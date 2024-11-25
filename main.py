from data import CIFDataset
from data import collate_pool, train_validate_test_loader
from cgcnn import CGCNN
from random import sample 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import time 
from torch.autograd import Variable
import numpy as np 
import sys 
import shutil

class Normalizer():
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std_dev = torch.std(tensor)
    
    def norm(self,tensor):
        return (tensor - self.mean)/self.std_dev
    
    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean
    
    def state_dict(self):
        return {
            'mean': self.mean,
            'std_dev': self.std_dev
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std_dev = state_dict['std_dev']

class AverageMeter():
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n 
        self.avg = self.sum/self.count 
def mae(prediction, target):
    return torch.mean(torch.abs(target - prediction))

def train(train_loader, model, criterion, optimizer, epoch, normalizer, gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if gpu:
            input_var = (Variable(input[0].cuda(non_blocking=True)),
                         Variable(input[1].cuda(non_blocking=True)),
                         input[2].cuda(non_blocking=True),
                         [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            input_var = (Variable(input[0]),
                         Variable(input[1]),
                         input[2],
                         input[3])
        target_normed = normalizer.norm(target)
        if gpu:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)
        
        #compute output 
        output = model(*input_var)
        loss = criterion(output, target_var)
        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        #compute gradient 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i%5 == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')


def validate(val_loader, model, criterion, normalizer, gpu):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.eval()
    end = time.time()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if gpu:
            with torch.no_grad():
                input_var = (Variable(input[0].cuda(non_blocking=True)),
                             Variable(input[1].cuda(non_blocking=True)),
                             input[2].cuda(non_blocking=True),
                             [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]])
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]),
                             Variable(input[1]),
                             input[2],
                             input[3])
                
        target_normed = normalizer.norm(target)
        if gpu:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)
        
        #compute output 
        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i%5==0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
    
    print(f'MAE: {mae_errors.avg:.3f}')
    return mae_errors.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def main(gpu=False):
    best_mae_error = 1e10
    cif_dir = '/content/drive/MyDrive/data/cif_files'
    atom_json_path = '/content/drive/MyDrive/data/atom_init.json'
    id_prop_path = '/content/drive/MyDrive/data/id_prop.csv'
    dataset = CIFDataset(cif_dir, atom_json_path, id_prop_path)
    collate_fn = collate_pool
    train_loader, test_loader, val_loader = train_validate_test_loader(
        dataset=dataset,
        batch_size=256,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        collate_fn=collate_fn,
        pin_memory=True if gpu else False
    )

    #creating a normalizer object 
    sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
    _, sample_target, _ = collate_pool(sample_data_list)
    normalizer = Normalizer(sample_target)

    #building Model 
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1]. shape[-1]
    model = CGCNN(orig_atom_fea_len, nbr_fea_len)

    if gpu:
        model.cuda()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 0.01)

    scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=[100])

    for epoch in range(30):
        train(train_loader, model, criterion, optimizer, epoch, normalizer, gpu)
        mae_error = validate(val_loader, model, criterion, normalizer, gpu)

        if np.isnan(mae_error):
            print('Exit due to NaN')
            sys.exit(1)
        
        scheduler.step()

        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_mae_error': best_mae_error,
            'optimizer': optimizer.state_dict(),
            'normalizer': normalizer.state_dict(),
        }, is_best)

    print('---------Evaluate Model on Test Set---------------')
    best_checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(best_checkpoint['state_dict'])
    validate(test_loader, model, criterion, normalizer, test=True)

if __name__ == '__main__':
    main(gpu=True)
        



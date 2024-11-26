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
import csv 

class Normalizer():
    def __init__(self, tensor):
        self.mean = torch.mean(tensor)
        self.std_dev = torch.std(tensor)
    
    def norm(self,tensor):
        return (tensor - self.mean)/self.std_dev
    
    def denorm(self, normed_tensor):
        return normed_tensor * self.std_dev + self.mean
    
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

def train(train_loader, model, criterion, optimizer, epoch, normalizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input_var = (Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3])
        target_normed = normalizer.norm(target)
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


def validate(val_loader, model, criterion, normalizer, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mae_errors = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    model.eval()
    end = time.time()

    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with torch.no_grad():
            input_var = (Variable(input[0]),
                        Variable(input[1]),
                        input[2],
                        input[3])
                
        target_normed = normalizer.norm(target)
        with torch.no_grad():
            target_var = Variable(target_normed)
        
        #compute output 
        output = model(*input_var)
        loss = criterion(output, target_var)

        mae_error = mae(normalizer.denorm(output.data.cpu()), target)
        losses.update(loss.data.cpu().item(), target.size(0))
        mae_errors.update(mae_error, target.size(0))

        if test:
            test_pred = normalizer.denorm(output.data.cpu())
            test_target = target
            test_preds += test_pred.view(-1).tolist()
            test_targets += test_target.view(-1).tolist()
            test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i%5==0:
            print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})')
    
    if test:
        with open('results/test_results.csv', 'w') as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets,
                                            test_preds):
                writer.writerow((cif_id, target, pred))
    
    print(f'MAE: {mae_errors.avg:.3f}')
    return mae_errors.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def mse(prediction, target):
    return torch.mean((target - prediction) ** 2)



def main():
    best_mae_error = 1e10
    cif_dir = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/cif_files'
    atom_json_path = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/atom_init.json'
    id_prop_path = '/Users/sparsh/Desktop/College core/CS_419/CGCNN_CS_419/data/id_prop.csv'
    dataset = CIFDataset(cif_dir, atom_json_path, id_prop_path)
    collate_fn = collate_pool
    train_loader, test_loader, val_loader = train_validate_test_loader(
        dataset=dataset,
        batch_size=256,
        train_ratio=0.75,
        val_ratio=0.125,
        test_ratio=0.125,
        collate_fn=collate_fn
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
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), 0.01)

    scheduler = MultiStepLR(optimizer, gamma=0.1, milestones=[50])

    for epoch in range(70):
        train(train_loader, model, criterion, optimizer, epoch, normalizer)
        mae_error = validate(val_loader, model, criterion, normalizer)

        if np.isnan(mae_error):
            print('Exit due to NaN')
            sys.exit(1)
        
        scheduler.step()
        
        is_best = mae_error < best_mae_error
        best_mae_error = min(mae_error, best_mae_error)

        with open('results/val_mae_err.txt', 'a') as f:
            f.write(f'Epoch: {epoch} -> MAE_err: {mae_error}\n')

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
    main()
        




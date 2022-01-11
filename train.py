import argparse
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import RecordUtil, build_dataset, build_model, test, train,prepare_datasets

def default_argument_parser():
    parser = argparse.ArgumentParser(description="image-super-resolution")
    parser.add_argument('--train-file', default=3,type=int) 
    parser.add_argument('--eval-file', default=3,type=int) 
    parser.add_argument('--batch-size', default=32,type=int) 
    parser.add_argument('--num-workers', default=4,type=int) 
    parser.add_argument('--lr', default=1e-4,type=float) 
    parser.add_argument('--epoch', default=100,type=int)   
    parser.add_argument('--f',default=5,type=int)
    parser.add_argument('--model-dir',default='./model',type=str)
    return parser

def main(args):
    
    args.train_file = f'./datasets/91-image_x{args.train_file}.h5'
    args.eval_file = f'./datasets/Set5_x{args.eval_file}.h5'
    #prepare_datasets(args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True

    model = build_model(args,device)
    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    loss_function = nn.MSELoss()
    train_dataloader,test_dataloader = build_dataset(args)

    EPOCH = args.epoch
    test_frequency = args.f

    epoch_loss = RecordUtil()
    epoch_psnr = RecordUtil()
    best_weights = copy.deepcopy(model.state_dict())
    best_psnr = 0
    best_epoch = 0

    for epoch in range(EPOCH):
        train(args,model,train_dataloader,optimizer,loss_function,epoch_loss,device,epoch)
        if epoch % test_frequency == 0:
            PSNR = test(model,test_dataloader,epoch_psnr,device,epoch)
            if PSNR>best_psnr:
                best_epoch = epoch
                best_psnr = PSNR
                best_weights = copy.deepcopy(model.state_dict())

    torch.save(best_weights, f'{args.model_dir}/best.pth')
    print('\n\n')
    print(f'best epoch = {best_epoch}')
    print(f'best psnr = {best_psnr}')
    print(f'best model weights was saved in {args.model_dir}/best.pth')
    print('-------------over-------------')



if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    main(args)
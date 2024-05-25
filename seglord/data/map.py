from .cityscape import CityNormal
from torch.utils.data import DataLoader

def get_data(args):
    if args.ds == 'citynormal':
        train_ds = CityNormal(root=args.dt, train=True, size=args.sz)
        valid_ds = CityNormal(root=args.dt, train=False, size=args.sz)

        args.inc = 3
        args.cls = 20
    
    train_dl = DataLoader(dataset=train_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)
    valid_dl = DataLoader(dataset=valid_ds, batch_size=args.bs, shuffle=True, pin_memory=args.pm, num_workers=args.wk)

    return train_dl, valid_dl, args
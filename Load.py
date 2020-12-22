import torch
import torchvision
import torchvision.transforms as transforms
import os.path


class DatasetTransformer(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)

def compute_mean_std(loader):
    # Compute the mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1

    return mean_img, std_img

def Normaliz(train_loader,valid_loader,test_loader):

    mean_train_tensor, std_train_tensor = compute_mean_std(train_loader)

    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean_train_tensor) / std_train_tensor)
    ])

    mean_valid_tensor, std_valid_tensor = compute_mean_std(valid_loader)

    valid_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean_valid_tensor) / std_valid_tensor)
    ])

    mean_test_tensor, std_test_tensor = compute_mean_std(test_loader)

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x - mean_test_tensor) / std_test_tensor)
    ])

    return train_transforms , valid_transforms , test_transforms

def Augment_Tronsform(Hflip=0.5,degrees=10,translate=(0.1, 0.1)):

    augment_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(Hflip),
        transforms.RandomAffine(degrees=degrees, translate=translate),
        transforms.ToTensor()
    ])
    return augment_transforms



def loadFashionMNIST(normaliz=False, augment_Tronsform=False):

        ############################################################################################ Datasets

        dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
        valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

        # Load the dataset for the training/validation sets
        train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                   train=True,
                                                   transform= None, #transforms.ToTensor(),
                                                   download=True)

        # Split it into training and validation sets
        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
        nb_valid =  int(valid_ratio * len(train_valid_dataset))
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


        # Load the test set
        test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                         transform= None, #transforms.ToTensor(),
                                                         train=False)


        train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
        valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())
        test_dataset = DatasetTransformer(test_dataset, transforms.ToTensor())

        ############################################################################################ Dataloaders
        num_threads = 4  # Loading the dataset is using 4 CPU threads
        batch_size = 128  # Using minibatches of 128 samples

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,  # <-- this reshuffles the data at every epoch
                                                   num_workers=num_threads)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=num_threads)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=num_threads)

        print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
        print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
        print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))



        if normaliz :
            train_transforms, valid_transforms, test_transforms = Normaliz(train_loader, valid_loader, test_loader)
            train_dataset = DatasetTransformer(train_dataset, train_transforms)
            valid_dataset = DatasetTransformer(valid_dataset, valid_transforms)
            test_dataset = DatasetTransformer(test_dataset, test_transforms)

        if augment_Tronsform :
            train_dataset = DatasetTransformer(train_dataset,Augment_Tronsform())
            valid_dataset = DatasetTransformer(valid_dataset,Augment_Tronsform())
            test_dataset = DatasetTransformer(test_dataset,Augment_Tronsform())

        return train_loader,valid_loader,test_loader
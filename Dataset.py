folder_lst = os.listdir("/content/drive/MyDrive/Data/METER_ML/")
train_count =0
val_count = 0
test_count = 0

for folder in folder_lst:
    print(folder+"train", len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/train_"+folder)))
    train_count += min(1000,len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/train_"+folder)))
    print(folder+"test", len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/test_"+folder)))
    test_count += len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/test_"+folder))
    print(folder+"val", len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/val_"+folder)))
    val_count+=len(os.listdir("/content/drive/MyDrive/Data/METER_ML/"+folder+"/val_"+folder))
print(train_count, val_count, test_count)
#train_count = 6000

val_count = 256

%pip install torchinfo



#Loading data into Transforms
from torchvision import datasets, transforms
from torch.utils.data import random_split
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomResizedCrop((224,224),scale=(0.8,1.0),ratio=(0.9,1.1)),
                                      transforms.ToTensor()
                                     ])
#dataset_size = train_count + val_count + test_count

def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes
classes = get_classes("/content/drive/MyDrive/Data/METER_ML/")
classes = ['CAFOs', 'Landfills', 'Mines', 'ProcessingPlants', 'RNT', 'WW']
classes

data_path = "/content/drive/MyDrive/Data/METER_ML/"

40/44, 22/40, 29/41, 18/19, 46/70, 21/42

# ['CAFOs', 'Landfills', 'Mines', 'Negative', 'ProcessingPlants', 'RNT', 'WW']
# cafo_list landfill_list mine_list  negative_list  pp_list  rnt_list  ww_list
#cafo_list = os.listdir(data_path+classes[0]+'/train_'+classes[0])[:1000]
'''cafo_list = os.listdir(data_path+classes[0]+'/train_'+classes[0])
landfill_list = os.listdir(data_path+classes[1]+'/train_'+classes[1])
mine_list = os.listdir(data_path+classes[2]+'/train_'+classes[2])
pp_list = os.listdir(data_path+classes[3]+'/train_'+classes[3])
rnt_list = os.listdir(data_path+classes[4]+'/train_'+classes[4])
ww_list = os.listdir(data_path+classes[5]+'/train_'+classes[5])
WWtrain 4635
WWtest 105
WWval 39
RNTtrain 1231
RNTtest 94
RNTval 50
ProcessingPlantstrain 588
ProcessingPlantstest 107
ProcessingPlantsval 38
Minestrain 598
Minestest 71
Minesval 39
Landfillstrain 1286
Landfillstest 95
Landfillsval 43
CAFOstrain 8191
CAFOstest 92
CAFOsval 47
5186 256 564
'''

cafo_list = os.listdir(data_path+classes[0]+'/train_'+classes[0])[6500:]
landfill_list = os.listdir(data_path+classes[1]+'/train_'+classes[1])
mine_list = os.listdir(data_path+classes[2]+'/train_'+classes[2])
pp_list = os.listdir(data_path+classes[3]+'/train_'+classes[3])
rnt_list = os.listdir(data_path+classes[4]+'/train_'+classes[4])
ww_list = os.listdir(data_path+classes[5]+'/train_'+classes[5])[3100:]

len(cafo_list), len(landfill_list)*1.25, len(mine_list)*2.5, len(pp_list)*2.5, len(rnt_list)*1.25,len(ww_list)

augmentation = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop((72,72)),
])
augmentation1 = transforms.Compose([
    transforms.RandomRotation(degrees=20),
    transforms.RandomVerticalFlip()
])
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
                                    ])

# cafo_list landfill_list mine_list  negative_list  pp_list  rnt_list  ww_list
dataset_size = len(cafo_list)+(int)(len(landfill_list)*1.25)+ (int)(len(mine_list)*2.5)+ (int)(len(pp_list)*2.5)+ (int)(len(rnt_list)*1.25)+len(ww_list)
#dataset_size = 9336
images = torch.zeros([dataset_size,15,72,72])
index = 0
flag = 0
target = torch.zeros((dataset_size))

for name in cafo_list:
  target[index] = 0
  imagePath = data_path+classes[0]+'/train_'+classes[0]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in landfill_list:
  flag+=1
  target[index] = 1
  imagePath = data_path+classes[1]+'/train_'+classes[1]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  temp = torch.nan_to_num(torch.Tensor(image_ar))
  images[index] = temp
  index+=1
  if flag%4 == 0:
    target[index] = 1
    images[index] = augmentation(temp)
    index+=1

for name in mine_list:
  flag+=1
  target[index] = 2
  imagePath = data_path+classes[2]+'/train_'+classes[2]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  temp = torch.nan_to_num(torch.Tensor(image_ar))
  images[index] = temp
  index+=1
  target[index] = 2
  images[index] = augmentation(temp)
  index+=1
  if flag%2==0:
    target[index] = 2
    images[index] = augmentation1(temp)
    index+=1

for name in pp_list:
  flag+=1
  target[index] = 3
  imagePath = data_path+classes[3]+'/train_'+classes[3]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  temp = torch.nan_to_num(torch.Tensor(image_ar))
  images[index] = temp
  index+=1
  target[index] = 3
  images[index] = augmentation(temp)
  index+=1
  if flag%2==0:
    target[index] = 3
    images[index] = augmentation1(temp)
    index+=1

for name in rnt_list:
  flag+=1
  target[index] = 4
  imagePath = data_path+classes[4]+'/train_'+classes[4]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  temp = torch.nan_to_num(torch.Tensor(image_ar))
  images[index] = temp
  index+=1
  if flag%4==0:
    target[index] = 4
    images[index] = augmentation(temp)
    index+=1

for name in ww_list:
  target[index] = 5
  imagePath = data_path+classes[5]+'/train_'+classes[5]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

#Image data stack
image_tens = torch.stack([images[i] for i in range(dataset_size)])
target_tens = torch.stack([target[i] for i in range(len(target))])
#Defining Custom dataset
data_set = torch.utils.data.TensorDataset(image_tens,target_tens)

train_loader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)

len(cafo_list)+(int)(len(landfill_list)*1.25)+ (int)(len(mine_list)*2.5)+ (int)(len(pp_list)*2.5)+ (int)(len(rnt_list)*1.25)+len(ww_list)

len(train_loader)

# Validation dataset

target = torch.zeros((val_count))
index = 0
for image in os.listdir(data_path+classes[0]+'/val_'+classes[0]):
  target[index] = 0
  index+=1
for image in os.listdir(data_path+classes[1]+'/val_'+classes[1]):
  target[index] = 1
  index+=1
for image in os.listdir(data_path+classes[2]+'/val_'+classes[2]):
  target[index] = 2
  index+=1
for image in os.listdir(data_path+classes[3]+'/val_'+classes[3]):
  target[index] = 3
  index+=1
for image in os.listdir(data_path+classes[4]+'/val_'+classes[4]):
  target[index] = 4
  index+=1
for image in os.listdir(data_path+classes[5]+'/val_'+classes[5]):
  target[index] = 5
  index+=1

target_tens = torch.stack([target[i] for i in range(len(target))])

cafo_list = os.listdir(data_path+classes[0]+'/val_'+classes[0])
landfill_list = os.listdir(data_path+classes[1]+'/val_'+classes[1])
mine_list = os.listdir(data_path+classes[2]+'/val_'+classes[2])
pp_list = os.listdir(data_path+classes[3]+'/val_'+classes[3])
rnt_list = os.listdir(data_path+classes[4]+'/val_'+classes[4])
ww_list = os.listdir(data_path+classes[5]+'/val_'+classes[5])

# cafo_list landfill_list mine_list  negative_list  pp_list  rnt_list  ww_list
images = torch.zeros([val_count,15,72,72])
index = 0
for name in cafo_list:
  imagePath = data_path+classes[0]+'/val_'+classes[0]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in landfill_list:
  imagePath = data_path+classes[1]+'/val_'+classes[1]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in mine_list:
  imagePath = data_path+classes[2]+'/val_'+classes[2]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in pp_list:
  imagePath = data_path+classes[3]+'/val_'+classes[3]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in rnt_list:
  imagePath = data_path+classes[4]+'/val_'+classes[4]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in ww_list:
  imagePath = data_path+classes[5]+'/val_'+classes[5]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

image_tens = torch.stack([images[i] for i in range(val_count)])

val_set_val = torch.utils.data.TensorDataset(image_tens,target_tens)
val_loader = torch.utils.data.DataLoader(val_set_val, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

dataloaders = {
    "train": train_loader,
    "val": val_loader
}
train_data_len = len(data_set)
valid_data_len = len(val_set_val)
dataset_sizes = {
    "train": train_data_len,
    "val": valid_data_len
}
print(len(train_loader), len(val_loader))

print(train_data_len, valid_data_len)


# Test dataset

target = torch.zeros((test_count))
index = 0
for image in os.listdir(data_path+classes[0]+'/test_'+classes[0]):
  target[index] = 0
  index+=1
for image in os.listdir(data_path+classes[1]+'/test_'+classes[1]):
  target[index] = 1
  index+=1
for image in os.listdir(data_path+classes[2]+'/test_'+classes[2]):
  target[index] = 2
  index+=1
for image in os.listdir(data_path+classes[3]+'/test_'+classes[3]):
  target[index] = 3
  index+=1
for image in os.listdir(data_path+classes[4]+'/test_'+classes[4]):
  target[index] = 4
  index+=1
for image in os.listdir(data_path+classes[5]+'/test_'+classes[5]):
  target[index] = 5
  index+=1

target_tens = torch.stack([target[i] for i in range(len(target))])

cafo_list = os.listdir(data_path+classes[0]+'/test_'+classes[0])
landfill_list = os.listdir(data_path+classes[1]+'/test_'+classes[1])
mine_list = os.listdir(data_path+classes[2]+'/test_'+classes[2])
pp_list = os.listdir(data_path+classes[3]+'/test_'+classes[3])
rnt_list = os.listdir(data_path+classes[4]+'/test_'+classes[4])
ww_list = os.listdir(data_path+classes[5]+'/test_'+classes[5])

# cafo_list landfill_list mine_list  negative_list  pp_list  rnt_list  ww_list
images = torch.zeros([test_count,15,72,72])
index = 0
for name in cafo_list:
  imagePath = data_path+classes[0]+'/test_'+classes[0]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in landfill_list:
  imagePath = data_path+classes[1]+'/test_'+classes[1]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in mine_list:
  imagePath = data_path+classes[2]+'/test_'+classes[2]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in pp_list:
  imagePath = data_path+classes[3]+'/test_'+classes[3]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)

  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in rnt_list:
  imagePath = data_path+classes[4]+'/test_'+classes[4]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

for name in ww_list:
  imagePath = data_path+classes[5]+'/test_'+classes[5]+'/'+name
  image_ar = rio.open(imagePath).read(out_shape=(15,72,72),resampling=Resampling.bilinear)
  images[index] = torch.nan_to_num(torch.Tensor(image_ar))

  index+=1

image_tens = torch.stack([images[i] for i in range(test_count)])

test_set = torch.utils.data.TensorDataset(image_tens,target_tens)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=4)

len(test_set)

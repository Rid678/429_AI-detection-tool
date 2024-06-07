import torch
import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import CocoDetection
import multiprocessing

# Define your custom collate function
def custom_collate(batch):
    images, targets = zip(*batch)
    images = list(images)  # Ensure images is a list of tensors
    return images, targets

# Define your custom dataset for training
class CustomCocoDataset(CocoDetection):
    def __init__(self, root, annFile, transform=None):
        super().__init__(root, annFile, transform)

    def __getitem__(self, idx):
        img, target_list = super().__getitem__(idx)

        # Process the list of annotations to extract bounding boxes and labels
        boxes = []
        labels = []
        for target in target_list:
            bbox = target['bbox']
            xmin, ymin, width, height = bbox
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(target['category_id'])

        target_modified = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }

        # Convert image to tensor if it's not already
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)

        return img, target_modified

# Define the path to your dataset and annotations JSON file
dataset_root = 'solenoid-valve-3/train/'
annotations_file = 'solenoid-valve-3/train/_annotations.coco.json'

# Create an instance of your custom dataset without any initial transform
custom_dataset = CustomCocoDataset(root=dataset_root, annFile=annotations_file)

# Define the dataloader for training with the custom collate function
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True, num_workers=multiprocessing.cpu_count(), collate_fn=custom_collate)

if __name__ == '__main__':
    # Define the Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # Modify the model's output layers to match the number of classes in your dataset
    # Include background as an additional class
    num_classes = 4  # 3 classes + 1 background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train the model
    num_epochs = 10  # Adjust the number of training epochs as needed
    for epoch in range(num_epochs):
        model.train()
        for images, targets in dataloader:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        lr_scheduler.step()

    # Save the trained model to a checkpoint file
    model_save_path = 'frcnn_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

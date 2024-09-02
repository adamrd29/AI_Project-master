from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, cuda, save, load, no_grad, max, amp, float32
from torch.optim import Adam
from tqdm import tqdm
from matplotlib import pyplot as plt
from time import sleep

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Flatten(),

            nn.Dropout(0.5),
            nn.LazyLinear(15),
            nn.Dropout(0.5),
            nn.LazyLinear(15),
        )

    def forward(self, x):
        return self.model(x)


def validate(device, image_classifier, data_loader):
    image_classifier.eval()
    criterion = nn.CrossEntropyLoss()
    loss = 0.0
    correct = 0
    total = 0
    with no_grad():
        for (images, labels) in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = image_classifier(images)
            loss += criterion(outputs, labels).item() * labels.size(0)

            _, prediction = max(outputs.data, 1)
            correct += (prediction == labels).sum()
            total += prediction.size(0)

    loss /= len(data_loader.dataset)
    accuracy = (correct / total) * 100
    image_classifier.train()
    return loss, accuracy


def train(device, image_classifier):
    transformations = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(float32, scale=True)
    ])

    training_data = ImageFolder(root='./archive/Training Data/Training Data', transform=transformations)

    # add augmented data to training set
    # augmented_data = ImageFolder(root='./archive/Train Augmented/Train Augmented', transform=transformations)
    # training_data = ConcatDataset([training_data, augmented_data])

    training_loader = DataLoader(training_data, batch_size=32, shuffle=True, pin_memory=(cuda.is_available()))

    valid_data = ImageFolder(root='./archive/Validation Data/Validation Data', transform=transformations)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(image_classifier.parameters(), lr=1e-3)
    scaler = cuda.amp.GradScaler()

    min_loss = float('inf')
    max_accuracy = 0
    no_progress = 0

    epochs = []
    training_loss_values = []
    validation_loss_values = []
    training_accuracy_values = []
    validation_accuracy_values = []

    for epoch in range(1, 100):
        running_loss = 0.0
        training_correct = 0
        training_total = 0
        for (images, labels) in tqdm(training_loader, desc=f'epoch {epoch}'):
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()

            with amp.autocast(device_type=device):
                outputs = image_classifier(images)
                loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()

            running_loss += (loss.item() * images.size(0))
            _, prediction = max(outputs.data, 1)
            training_correct += (prediction == labels).sum()
            training_total += prediction.size(0)

        training_loss = running_loss / len(training_loader.dataset)
        training_accuracy = (training_correct / training_total) * 100
        validation_loss, validation_accuracy = validate(device, image_classifier, valid_loader)

        print(f'training loss: {training_loss} | validation loss: {validation_loss} | training accuracy: {training_accuracy:0.2f}% | validation accuracy: {validation_accuracy:0.2f}%')

        epochs.append(epoch)
        training_loss_values.append(training_loss)
        validation_loss_values.append(validation_loss)
        training_accuracy_values.append(training_accuracy.cpu())
        validation_accuracy_values.append(validation_accuracy.cpu())

        if (validation_loss < min_loss) and (validation_accuracy > max_accuracy):
            min_loss = validation_loss
            save(image_classifier.state_dict(), './min_loss.pt')
            max_accuracy = validation_accuracy
            save(image_classifier.state_dict(), './max_accuracy.pt')
            no_progress = 0
            print('min_loss.pt and max_accuracy.pt updated')
        else:
            if validation_loss < min_loss:
                min_loss = validation_loss
                save(image_classifier.state_dict(), './min_loss.pt')
                print('min_loss.pt updated')

            if validation_accuracy > max_accuracy:
                max_accuracy = validation_accuracy
                save(image_classifier.state_dict(), './max_accuracy.pt')
                print('max_accuracy.pt updated')
                no_progress = 0
            else:
                no_progress += 1

        if no_progress == 5:
            break

        print('')
        sleep(0.1)

    plt.subplot(121)
    plt.plot(epochs, training_loss_values, label='Training Loss')
    plt.plot(epochs, validation_loss_values, label='Validation Loss')
    plt.xticks(epochs[1::2])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(epochs, training_accuracy_values, label='Training Accuracy')
    plt.plot(epochs, validation_accuracy_values, label='Validation Accuracy')
    plt.xticks(epochs[1::2])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()


def test(device, image_classifier):
    transformations = v2.Compose([
        v2.Resize([256, 256], antialias=True),
        v2.ToImage(),
        v2.ToDtype(float32, scale=True)
    ])
    testing_data = ImageFolder(root='./archive/Testing Data/Testing Data', transform=transformations)
    testing_loader = DataLoader(testing_data, batch_size=32, shuffle=True, pin_memory=True)

    image_classifier.load_state_dict(load('./max_accuracy.pt'))
    loss, accuracy = validate(device, image_classifier, testing_loader)
    print(f'max_accuracy.pt | loss: {loss} | accuracy: {accuracy:0.2f}%')

    image_classifier.load_state_dict(load('./min_loss.pt'))
    loss, accuracy = validate(device, image_classifier, testing_loader)
    print(f'min_loss.pt | loss: {loss} | accuracy: {accuracy:0.2f}%')


def main():
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'running on {device}')
    image_classifier = ImageClassifier().to(device)

    train(device, image_classifier)
    test(device, image_classifier)


if __name__ == '__main__':
    main()

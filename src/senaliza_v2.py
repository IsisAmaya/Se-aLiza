"""
Este módulo realiza la clasificación de gestos de la mano en lenguaje de señas colombiano
usando PyTorch y un modelo preentrenado ResNet50.
"""

from torch import nn
import torch
import torch.nn.functional as F
from torchvision import models, transforms as T
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


# Directorio de datos
DATA_DIR = "data/WB_LSC"

# Aplicar transformaciones de preprocesamiento de datos
train_tfms = T.Compose([
    T.RandomCrop(258, padding=4, padding_mode="reflect"),
    T.Resize(64),
    T.RandomVerticalFlip(),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

# Conjunto de datos temporal
train_ds_temp = ImageFolder(DATA_DIR, train_tfms)

# Dividir en conjunto de entrenamiento y validación
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

VAL_SIZE = 200
TRAIN_SIZE = len(train_ds_temp) - VAL_SIZE

train_ds, valid_ds = random_split(train_ds_temp, [TRAIN_SIZE, VAL_SIZE])

# Verificar tamaños de los datasets
print(len(train_ds), len(valid_ds))

# Creación de DataLoader
train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds, batch_size=64, shuffle=True, num_workers=0, pin_memory=True
)

print(len(train_dl), len(valid_dl))

def accuracy(outputs, labels):
    """Calcula la precisión de las predicciones."""
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ColombianHandGestureImageClassificationBase(nn.Module):
    """Clase base para la clasificación de imágenes."""
    def forward(self, input_batch):
        """Define el forward pass en la clase base."""
        raise NotImplementedError("La subclase debe implementar este método")

    def training_step(self, batch):
        """Realiza una etapa de entrenamiento."""
        images, targets = batch
        out = self(images)
        loss = F.cross_entropy(out, targets)
        return loss

    def validation_step(self, batch):
        """Realiza una etapa de validación."""
        images, targets = batch
        out = self(images)
        loss = F.cross_entropy(out, targets)
        acc = accuracy(out, targets)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        """Procesa los resultados al final de la época de validación."""
        batch_losses = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        """Imprime los resultados de la época."""
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.4f}, "
              f"train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, "
              f"val_acc: {result['val_acc']:.4f}")

class ColombianHandGestureResnet(ColombianHandGestureImageClassificationBase):
    """Modelo basado en ResNet50 para la clasificación de gestos de la mano."""
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 22)

    def forward(self, input_batch):
        """Define el forward pass del modelo."""
        return torch.sigmoid(self.network(input_batch))

    def freeze(self):
        """Congela las capas del modelo excepto la última capa totalmente conectada."""
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True

    def unfreeze(self):
        """Descongela todas las capas del modelo."""
        for param in self.network.parameters():
            param.require_grad = True

# Funciones para manejar dispositivos y datos

def get_default_device():
    """Elige GPU si está disponible, de lo contrario CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def to_device(data, device):
    """Mueve tensor(es) al dispositivo seleccionado."""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Envuelve un DataLoader para mover datos a un dispositivo."""
    def __init__(self, data_loader, device_local):
        self.data_loader = data_loader
        self.device_local = device_local

    def __iter__(self):
        """Devuelve un lote de datos después de moverlos al dispositivo."""
        for batch in self.data_loader:
            yield to_device(batch, self.device_local)

    def __len__(self):
        """Número de lotes."""
        return len(self.data_loader)

# Configuración de dispositivos
device_global = get_default_device()

train_dl = DeviceDataLoader(train_dl, device_global)
valid_dl = DeviceDataLoader(valid_dl, device_global)

@torch.no_grad()
def evaluate(model_eval, val_loader):
    """Evalúa el modelo con el conjunto de validación."""
    model_eval.eval()
    outputs = [model_eval.validation_step(batch) for batch in val_loader]
    return model_eval.validation_epoch_end(outputs)

def get_lr(optimizer):
    """Obtiene la tasa de aprendizaje actual del optimizador."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]

def fit_one_cycle(config_dict, train_model, train_loader, val_loader):
    """Entrena el modelo utilizando el ciclo de aprendizaje de una sola pasada."""
    torch.cuda.empty_cache()
    history_log = []

    optimizer = config_dict['opt_func'](
    train_model.parameters(),
    config_dict['max_lr'],
    weight_decay=config_dict['weight_decay']
    )

    sched = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, config_dict['max_lr'], epochs=config_dict['epochs'],
    steps_per_epoch=len(train_loader)
    )


    for epoch in range(config_dict['epochs']):
        # Fase de entrenamiento
        train_model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = train_model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Clipping de gradiente
            if config_dict['grad_clip']:
                nn.utils.clip_grad_value_(train_model.parameters(), config_dict['grad_clip'])

            optimizer.step()
            optimizer.zero_grad()

            # Grabar y actualizar la tasa de aprendizaje
            lrs.append(get_lr(optimizer))
            sched.step()

        # Fase de validación
        result = evaluate(train_model, val_loader)
        result["train_loss"] = torch.stack(train_losses).mean().item()
        result["lrs"] = lrs
        train_model.epoch_end(epoch, result)
        history_log.append(result)
    return history_log

# Entrenar el modelo

model = to_device(ColombianHandGestureResnet(), device_global)

history = [evaluate(model, valid_dl)]


model.freeze()

config = {
    'epochs': 15,
    'max_lr': 0.001,
    'grad_clip': 0.1,
    'weight_decay': 1e-4,
    'opt_func': torch.optim.Adam
}

# Descomentar para entrenar el modelo
# history = fit_one_cycle(config, model, train_dl, valid_dl)

# Guardar el modelo
# torch.save(model, "senalizav2.pt")

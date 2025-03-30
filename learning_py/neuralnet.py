import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, train_test_split
import numpy as np


class SiameseNetworkDataset(Dataset):
    def __init__(self, data_list):

        self.data_list = data_list

    def __getitem__(self, index):
        vec1_full, vec2_full, score = self.data_list[index]
        vec1_full = torch.from_numpy(vec1_full).float()
        vec2_full = torch.from_numpy(vec2_full).float()
        score_t   = torch.tensor([score], dtype=torch.float32)
        return (vec1_full, vec2_full, score_t)

    def __len__(self):
        return len(self.data_list)


class SiameseNetworkFusion(nn.Module):
    def __init__(self):
        super().__init__()

 
        self.num_points_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.area_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )
        self.centroid_layer = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.bbox_layer = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.compactness_layer = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU()
        )

     
        self.fusion_layer = nn.Sequential(
            nn.Linear(36, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

    def encode_9d(self, x):
        num_points  = x[:, 0:1]
        area        = x[:, 1:2]
        centroid    = x[:, 2:4]
        bbox        = x[:, 4:8]
        compactness = x[:, 8:9]

        np_emb   = self.num_points_layer(num_points)
        area_emb = self.area_layer(area)
        ctr_emb  = self.centroid_layer(centroid)
        bbox_emb = self.bbox_layer(bbox)
        comp_emb = self.compactness_layer(compactness)

        combined = torch.cat([np_emb, area_emb, ctr_emb, bbox_emb, comp_emb], dim=1)
        fused_emb = self.fusion_layer(combined)
        return fused_emb

    def forward(self, x1, x2):
        emb1 = self.encode_9d(x1)
        emb2 = self.encode_9d(x2)
        return emb1, emb2


class ClampedDistanceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, emb1, emb2, label):
        dist = torch.norm(emb1 - emb2, p=2, dim=1).unsqueeze(1)
        pred_dist = dist / (1.0 + dist)
        label = label.view(-1, 1)
        loss = self.mse_loss(pred_dist, label)
        return loss


class SiameseModelTrainerFusion:
    def __init__(self):
        self.model = None
        self.criterion = None
        self.optimizer = None

    def train_model(self, data_list, epochs=50, patience=10):
        dataset = SiameseNetworkDataset(data_list)

        train_indices, test_indices = train_test_split(
            range(len(dataset)),
            test_size=0.1,
            shuffle=True,
            random_state=42
        )
        train_dataset = Subset(dataset, train_indices)
        test_dataset  = Subset(dataset, test_indices)

        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
        weight_decays  = [0.0, 1e-4]  
        best_avg_val_loss = float('inf')
        best_hyperparams  = None

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)

        for lr in learning_rates:
            for wd in weight_decays:
                fold_val_losses = []

                for fold, (cv_train_ids, cv_val_ids) in enumerate(kfold.split(train_dataset)):
                    fold_train_loader = DataLoader(
                        Subset(train_dataset, cv_train_ids),
                        batch_size=24,
                        shuffle=True
                    )
                    fold_val_loader = DataLoader(
                        Subset(train_dataset, cv_val_ids),
                        batch_size=24,
                        shuffle=False
                    )

  
                    model = SiameseNetworkFusion()
                    criterion = ClampedDistanceLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

                    best_val_loss = float('inf')
                    epochs_no_improve = 0

          
                    for epoch in range(epochs):
                        model.train()
                        total_loss = 0.0

                        for batch_idx, (vec1, vec2, scores) in enumerate(fold_train_loader):
                            optimizer.zero_grad()
                            emb1, emb2 = model(vec1, vec2)
                            loss = criterion(emb1, emb2, scores)
                            loss.backward()
                            optimizer.step()
                            total_loss += loss.item()

                        
                        avg_train_loss = total_loss / len(fold_train_loader)

                    
                        model.eval()
                        val_loss = 0.0
                        with torch.no_grad():
                            for vec1, vec2, scores in fold_val_loader:
                                emb1, emb2 = model(vec1, vec2)
                                val_loss += criterion(emb1, emb2, scores).item()
                        val_loss /= len(fold_val_loader)

                      
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                            if epochs_no_improve >= patience:
                                break

                    fold_val_losses.append(best_val_loss)

                
                avg_val_loss = np.mean(fold_val_losses)

           
                if avg_val_loss < best_avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_hyperparams  = {'lr': lr, 'weight_decay': wd}

       
        print(f"Best hyperparams from CV: {best_hyperparams}, val_loss={best_avg_val_loss:.4f}")
        self.model = SiameseNetworkFusion()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=best_hyperparams['lr'],
            weight_decay=best_hyperparams['weight_decay']
        )
        self.criterion = ClampedDistanceLoss()

        full_train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            for batch_idx, (vec1, vec2, scores) in enumerate(full_train_loader):
                self.optimizer.zero_grad()
                emb1, emb2 = self.model(vec1, vec2)
                loss = self.criterion(emb1, emb2, scores)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(full_train_loader)
            # Debug print
            # print(f"Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_loss:.4f}")

   
        test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for vec1, vec2, scores in test_loader:
                emb1, emb2 = self.model(vec1, vec2)
                test_loss += self.criterion(emb1, emb2, scores).item()
        test_loss /= len(test_loader)

        print(f"Final Test Loss: {test_loss:.4f}")
        return self.model

    def find_most_similar(self, input_vector, repo_vectors):
        self.model.eval()

        input_tensor = torch.from_numpy(input_vector).float().unsqueeze(0)  # (1, 9)

        with torch.no_grad():
            emb_input = self.model.encode_9d(input_tensor)

            repo_tensor = torch.from_numpy(np.array(repo_vectors)).float()  # (N, 9)
            emb_repo = self.model.encode_9d(repo_tensor)

            dists = torch.norm(emb_input - emb_repo, p=2, dim=1)

            pred_dists = dists / (1.0 + dists)

            min_index = torch.argmin(pred_dists).item()
            min_distance = pred_dists[min_index].item()
        return min_index, min_distance




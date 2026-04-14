## Model Hyperparameters

| Model  | Dataset           | Script Hyperparameters |
|--------|------------------|----------------------|
| GRU    | MIMIC-III        | hid_dim = 50, dropout = 0.2, lr = 0.0001 |
|        | PhysioNet-2012   | hid_dim = 43, dropout = 0.2, lr = 0.0001 |
| TCN    | MIMIC-III        | hid_dim = 128, dropout = 0.1, lr = 0.0001 |
|        | PhysioNet-2012   | hid_dim = 64, dropout = 0.1, lr = 0.0005 |
| SaND   | MIMIC-III        | hid_dim = 64, dropout = 0.3, attention_dropout = 0.3, lr = 0.0005 |
|        | PhysioNet-2012   | hid_dim = 64, dropout = 0.3, attention_dropout = 0.3, lr = 0.0005 |
| GRU-D  | MIMIC-III        | hid_dim = 60, dropout = 0.2, lr = 0.0001 |
|        | PhysioNet-2012   | hid_dim = 49, dropout = 0.2, lr = 0.0001 |
| STraTS | MIMIC-III        | hid_dim = 50, dropout = 0.2, lr = 0.0005 |
|        | PhysioNet-2012   | hid_dim = 50, dropout = 0.2, lr = 0.0005 |

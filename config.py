class Config:
  def __init__(self, epochs=3, 
                    lr=1.1e-5, 
                    batch_size=32, 
                    model='roberta-base', 
                    seq_length=256):
    self.epochs = epochs
    self.lr = lr
    self.batch_size = batch_size
    self.model = model
    self.seq_length = seq_length
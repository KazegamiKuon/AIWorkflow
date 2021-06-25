from torch import nn
from ..utils.activations import get_activation

class ElectraDiscriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config
    
    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits

# loss for generator and discriminator
# loss_fct = nn.BCEWithLogitsLoss()
# loss = loss_fct(discriminator_output.view(-1, generator_output.shape[1]), labels.float())


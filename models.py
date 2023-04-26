import torch
#import open_clip
from torch import nn
import timm



class ImageEncoder(nn.Module):
      def __init__(self,model_name,pretrained=True,trainable=True):
          super().__init__()
          self.model = timm.create_model(
            model_name, pretrained, num_classes=2, global_pool="avg") 
          for p in self.model.parameters():
            p.requires_grad = trainable

          def forward(self, x):
              return self.model(x)



class TextEncoder(nn.Module):
      def __init__(self,model_name,pretrained=True,trainable=True):
          super().__init__()
          if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
          else:
            self.model = DistilBertModel(config=DistilBertConfig())
          for p in self.model.parameters():
            p.requires_grad = trainable
          self.target_token_idx=0
      def forward(self, input_ids, attention_mask):
          output = self.model(input_ids=input_ids, attention_mask=attention_mask)
          last_hidden_state = output.last_hidden_state
          return last_hidden_state[:, self.target_token_idx, :]     



class Projection(nn.Module):
      def __init__(self,embedding_dim,projection_dim,dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout=nn.Dropout(dropout)
     def forward(self, x):
         projected = self.projection(x)
         x = self.gelu(projected)
         x = self.fc(x)
         x = self.dropout(x)
         x = x + projected
         x = self.layer_norm(x)
         return x


class CLIPModel(nn.Module):
      def __init__(self,temperature=args.temperature,image_embedding=args.image_embedding,text_embedding=args.text_embedding):
          super().__init__()
          self.image_encoder = ImageEncoder()
          self.text_encoder = TextEncoder()
          self.image_projection = ProjectionHead(embedding_dim=image_embedding)
          self.text_projection = ProjectionHead(embedding_dim=text_embedding)
          self.temperature = args.temperature

      def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


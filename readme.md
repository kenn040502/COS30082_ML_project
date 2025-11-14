1. Backbone: DINOv2 feature extractor

You use vit_base_patch14_reg4_dinov2.lvd142m from timm.
This model is pre-trained on a huge generic dataset, so it already knows a lot about shapes, textures, colours, etc.
In your code, this backbone is frozen (not updated).
It just turns each image into a feature vector (like a 512-D description of the plant).
Think of DINOv2 as a smart “image to features” machine that you reuse instead of training from scratch.

2. New head on top (transfer learning)

On top of the DINO features, you add a small classification head:
a couple of fully-connected layers that end in 100-class softmax (since you have 100 species).
This head is trained on your dataset (herbarium + photo), while DINO stays fixed.
Loss: main loss is cross-entropy, with class weights to handle imbalance.
So this part is standard supervised transfer learning:
“Use pre-trained DINO features, learn a new head for my 100 classes.”

3. Triplet loss = metric learning on the embedding

Besides cross-entropy, you also compute a small triplet loss on the embedding.

Idea:

For each anchor image, pull same-class images closer.
Push different-class images further away.
This makes the embedding space more “clustered by species”, not just good for softmax.

So you are combining:
Classification loss (CE) + metric learning loss (triplet).

That’s why I’d describe the method as:
“DINOv2 transfer learning with joint classification + triplet metric learning.”

4. Domain aspect (herbarium vs photo)
Training uses both domains (herbarium and photo) mixed together.
We don’t do heavy domain-adaptation tricks (like GRL, CDAN, etc.).
Instead, we rely on:
strong backbone (DINOv2)
metric learning (triplet)
to help the model generalise across style differences between herbarium and field photos.

Later, if you add pairs.csv, the same framework will let you measure performance separately on:

classes that have paired herbarium–photo data
classes that don’t have pairs
But method-wise, what you’re using now is:

💡 Approach 3: DINOv2-based supervised transfer learning with auxiliary triplet loss for cross-domain plant species classification.
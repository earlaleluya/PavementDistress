from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch


class ViTClassifier:
    """
    A Vision Transformer (ViT) based image classifier that uses a pre-trained model
    for image classification tasks. The classifier supports both CPU and GPU devices.

    Attributes:
        device (str): The device to run the model on ('cuda' or 'cpu').
        extractor (ViTFeatureExtractor): The feature extractor for preprocessing input images.
        model (ViTForImageClassification): The pre-trained ViT model for image classification.

    Note:
        Input images should be provided as tensors of shape (H, W, C), where H is the height,
        W is the width, and C is the number of channels (typically 3 for RGB images). The dtype
        of the input images must be `torch.uint8`.

    Example:
        >>> classifier = ViTClassifier(model_path="google/vit-base-patch16-224", device="cuda")
        >>> from PIL import Image
        >>> import torchvision.transforms as transforms
        >>> transform = transforms.ToTensor()
        >>> image = transform(Image.open("example.jpg")).permute(1, 2, 0).mul(255).byte()  # Convert to (H, W, C) and uint8
        >>> outputs = classifier.classify([image])
        >>> print(outputs)
    """

    
    
    def __init__(self, model_path, device=None):
        """
        Initializes the ViTClassifier with a pre-trained model.
     
        Args:
            model_path (str): Path to the pre-trained ViT model.
            device (str, optional): The device to run the model on. Defaults to 'cuda' if available, otherwise 'cpu'.
        
        Raises:
            ValueError: If the model_path is invalid or the model cannot be loaded.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.extractor = ViTFeatureExtractor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path).to(device=self.device)
        self.model.eval()




    def classify(self, images):
        """
        Classifies a batch of input images.

        Args:
            images (list of torch.Tensor): A list of input images as PyTorch tensors.
       
         Returns:
            list of list of dict: A list where each element corresponds to an image and contains a list of
            dictionaries with classification labels and their associated probabilities, sorted in descending
            order of probability.

            Example:
                [
                    [
                        {"label": "thincrack", "score": 0.9747543334960938},
                        {"label": "sealedcrack", "score": 0.021938998252153397},
                        {"label": "sealedpatch", "score": 0.0031411931850016117},
                        {"label": "nondistressed", "score": 0.00016548742132727057},
                    ],
                    ...
                ]
        """
        inputs = self.transform(images)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        results = []
        for prob in probs:
            labels_and_scores = [
                {"label": self.model.config.id2label[idx], "score": score.item()}
                for idx, score in enumerate(prob)
            ]
            sorted_labels_and_scores = sorted(labels_and_scores, key=lambda x: x["score"], reverse=True)
            results.append(sorted_labels_and_scores)
        return results




    def transform(self, images):
        """
        Transforms input images into the format required by the ViT model.

        Args:
            images (list of np.ndarrayy): A list of input images as Numpy arrays.
            
        Returns:
            dict: A dictionary containing the preprocessed images as tensors, ready for input to the model.
        """
        return self.extractor(images=images, return_tensors="pt").to(device=self.device)
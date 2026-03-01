Uses the Metropolitan Museum of Art’s Open Access API, the platform analyzes high-resolution imagery to predict the creation date of a painting based on technical patterns that the human eye might miss.

Features: 

Color Theory Analytics: Using K-Means clustering, the system isolates the dominant color palettes of a piece. 
It then maps these against HSV (Hue, Saturation, Value) distributions and luminance statistics to identify era-specific pigment trends.

Structural Texture: To solve the challenge of identifying brushwork, the pipeline utilizes Sobel edge detection and Laplacian operators. 
These filters allow the model to "feel" the canvas, distinguishing between the near-invisible transitions of Renaissance sfumato and the heavy, physical impasto of the Modern era.

Statistical Composition: The model calculates the skewness and kurtosis of color channels and luminance, capturing the specific way light and shadow were manipulated during periods like the Baroque.

Model Architecture and Inference
The engine evaluates the data through an ensemble of Gradient Boosting and Random Forest regressors, choosing the optimal model based on the lowest Mean Absolute Error during training.

To ensure the system isn't fooled by poor lighting or low-resolution uploads, it performs a five-pass inference cycle. It adjusts the contrast and brightness of the input image and re-analyzes it multiple times.
Ultimately producing a weighted ensemble prediction. This results in a "Confidence Score" that informs the user how reliable the prediction is based on the visible detail of the brushstrokes and color variance.

import tensorflow as tf

def VGG(content_image):
    x = tf.keras.applications.vgg19.preprocess_input(content_image * 255)
    x = tf.image.resize(x, (224, 224))
    vgg = tf.keras.applications.VGG19(include_top=True, weights='imagenet')
    prediction_probabilities = vgg(x)
    print(prediction_probabilities.shape)
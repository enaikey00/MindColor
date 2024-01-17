## --- Titolo Progetto ---

## --- Imports and Libraries ---

import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import time
from IPython import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

  ## --- Dataset ---
  # Load Images
  images_dataset = image_dataset_from_directory(
      directory="./data/images/",
      labels=None,
      color_mode="rgb",
      batch_size=8,
      image_size=(28, 28),
      shuffle=True,
      seed=10
  )
  BATCH_SIZE = 8
  print(images_dataset)

  # Load Brain Waves
  BUFFER_SIZE = 1500 * 2 # about 1500 samples for each image class
  colnames = ["α_wave", "β_wave", "δ_wave", "θ_wave", "γ_wave"]
  df = pd.read_csv("./data/waves/final_df.csv", names=colnames) # will be rosso_blue_green_yellow.csv
  print(df.head())
  waves_dataset = tf.data.Dataset.from_tensor_slices(tf.constant(df.values))
  waves_dataset = waves_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  for e in waves_dataset.take(1):
    print(e)
  print(waves_dataset.take(1))
  print()

  ## --- Generator ---
  def make_generator_model():
      model = tf.keras.Sequential()
      model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(5,)))
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Reshape((7, 7, 256)))
      assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

      model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
      assert model.output_shape == (None, 7, 7, 128)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
      assert model.output_shape == (None, 14, 14, 64)
      model.add(layers.BatchNormalization())
      model.add(layers.LeakyReLU())

      #model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
      # tanh scales the output to the range [-1, 1], which is suitable for grayscale images,
      # for RGB images, scale the values to the range [0, 1] with sigmoid
      model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='sigmoid'))
      assert model.output_shape == (None, 28, 28, 3) # 3 for our use case

      return model

  generator = make_generator_model()

  # Example
  #noise = tf.random.normal([1, 100]) # this will be a sample brain wave
  #generated_image = generator(noise, training=False)

  #plt.imshow(generated_image[0, :, :, 0], cmap='gray')

  ## --- Discriminator ---

  def make_discriminator_model():
      model = tf.keras.Sequential()
      model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 3]))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
      model.add(layers.LeakyReLU())
      model.add(layers.Dropout(0.3))

      model.add(layers.Flatten())
      model.add(layers.Dense(1))

      return model

  discriminator = make_discriminator_model()

  # Example
  #decision = discriminator(generated_image)
  #print(decision)

  ### --- Loss and Optimizers ---
  # This method returns a helper function to compute cross entropy loss
  cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

  # Discriminator Loss
  def discriminator_loss(real_output, fake_output):
      real_loss = cross_entropy(tf.ones_like(real_output), real_output)
      fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
      total_loss = real_loss + fake_loss
      return total_loss

  # Generator Loss
  def generator_loss(fake_output):
      return cross_entropy(tf.ones_like(fake_output), fake_output)
      
  # Optimizers
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  ## --- Save Checkpoint ---
  checkpoint_dir = './training_checkpoints_v2'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                  discriminator_optimizer=discriminator_optimizer,
                                  generator=generator,
                                  discriminator=discriminator)

  ## --- Train Loop ---
  # Constants
  EPOCHS = 50
  noise_dim = 5 # this will be "brain waves dimension"
  num_examples_to_generate = 16

  # You will reuse this seed overtime (so it's easier)
  # to visualize progress in the animated GIF)
  seed = tf.random.normal([num_examples_to_generate, noise_dim])

  # Notice the use of `tf.function`
  # This annotation causes the function to be "compiled".
  @tf.function
  def train_step(images, noise):
      #noise = tf.random.normal([BATCH_SIZE, noise_dim]) # next(generator_function()) batch_size x 5 (numero di frequenze)
      print(noise)

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

      # calcolo derivata funzione di costo rispetto alle variabili
      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables) # le variabili sono gli input o i pesi?
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      # applicazione modifiche ai pesi della rete
      generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
      return gen_loss, disc_loss
      
  gen_losses = []
  disc_losses = []

  def train(dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for image_batch, wave_batch in zip(dataset, waves_dataset): # try this without the dataset_gen; change generator input shape to (5,)
        noise = wave_batch
        gen_loss, disc_loss = train_step(image_batch, noise)
      
        # append losses to the lists
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())

      # Produce images for the GIF as you go
      #display.clear_output(wait=True)
      #generate_and_save_images(generator,
      #                         epoch + 1,
      #                         seed)

      # Save the model every 15 epochs
      if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
      print('Generator Loss: {}, Discriminator Loss: {}'.format(gen_losses[-1], disc_losses[-1]))

      # Plot losses at each epoch
      #plot_losses(gen_losses, disc_losses)
    return gen_losses, disc_losses

    # Generate after the final epoch
    #display.clear_output(wait=True)
    #generate_and_save_images(generator,
    #                         epochs,
    #                         seed)

  ## --- Generate and Save Images ---
  def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    print()
    print("Predictions: ", predictions)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, -1])
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

  ## --- Plot losses ---
  def plot_losses(gen_losses, disc_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(gen_losses, label='Generator Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

  ## --- Train model ---
  gen_losses, disc_losses = train(images_dataset, EPOCHS)
  plot_losses(gen_losses, disc_losses)

  ## --- Inference ---
  for batch in waves_dataset:
    generate_and_save_images(model = generator, epoch = 1, test_input = batch)
    break

  ## --- Save Models ---
  generator.save('./models/generator.keras')
  discriminator.save('./models/discriminator.keras')




  ## --- Evaluate Model ---

  ## --- Test Set ---
  # Load Images
  images_testset = image_dataset_from_directory(
      directory="./data/test/images/",
      labels=None,
      color_mode="rgb",
      batch_size=8,
      image_size=(28, 28),
      shuffle=True,
      seed=10
  )
  BATCH_SIZE = 8

  # Load Brain Waves
  BUFFER_SIZE = 1500 * 2 # about 1500 samples for each image class
  colnames = ["α_wave", "β_wave", "δ_wave", "θ_wave", "γ_wave"]
  df = pd.read_csv("./data/test/waves/df.csv", names=colnames) # will be rosso_blue_green_yellow.csv
  print(df.head())
  waves_testset = tf.data.Dataset.from_tensor_slices(tf.constant(df.values))
  waves_testset = waves_testset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


  @tf.function
  def eval_step(images, noise):

      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=False)

        real_output = discriminator(images, training=False)
        fake_output = discriminator(generated_images, training=False)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        return gen_loss, disc_loss

  gen_losses = []
  disc_losses = []

  def evaluate(dataset, epochs):
    for epoch in range(epochs):
      start = time.time()

      for image_batch, wave_batch in zip(dataset, waves_testset):
        noise = wave_batch
        gen_loss, disc_loss = eval_step(image_batch, noise)
      
        # append losses to the lists
        gen_losses.append(gen_loss.numpy())
        disc_losses.append(disc_loss.numpy())

      # Save the model every 15 epochs
      if (epoch + 1) % 15 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
      print('Generator Loss: {}, Discriminator Loss: {}'.format(gen_losses[-1], disc_losses[-1]))

      # Plot losses at each epoch
      #plot_losses(gen_losses, disc_losses)
    return gen_losses, disc_losses


  # Evaluate
  gen_loss, disc_loss = evaluate(images_testset, 20)
  plot_losses(gen_losses, disc_losses)

  # Inference of Images on test set
  for batch in waves_testset:
    generate_and_save_images(model = generator, epoch = 1, test_input = batch)
    break
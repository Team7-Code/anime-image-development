import keras
from keras.engine.saving import load_model
from keras.optimizers import Adam

from discriminator import *
from discriminator import discriminator
from generator import *
from generator import generator
from metrics import metrics
from plots import *
from readImages import *

model_path=""
model_checkpoint = input("Do you want to use a saved model for training? Y for Yes and N for No: ")
model_checkpoint = model_checkpoint.lower()
if model_checkpoint == 'y':
    model_path = input("Enter model path: ")
imagePath = input("Enter the path for images: ")

numberOfImages = input("How many images to train with? max=14310: ")
steps = input("How many steps to train for? ")

np.random.seed(0)
gen_path = "gen\\"

with open('keep_count.txt', 'r') as f:
   run_count = f.read()

if not os.path.exists(gen_path + "run time " + str(run_count)):
   os.mkdir(gen_path + "run time " + str(run_count))
store_path = (gen_path + "run time " + str(run_count))

num_residual_blocks = 16

LRimagesPath = imagePath
HRimagesPath = imagePath
# LRimagesPath = "..\\animeChars\\moeimouto-faces"
# HRimagesPath = "..\\animeChars\\moeimouto-faces"

LRDim = 32
HRDim = 128

LRimages = readImagesShort(LRimagesPath, count=int(numberOfImages), scale_size=LRDim)
print(LRimages.shape)
HRimages = readImagesShort(HRimagesPath, count=int(numberOfImages), scale_size=HRDim)
print(HRimages.shape)

LRimages = LRimages.astype('float32')
HRimages = HRimages.astype('float32')

LRimages = LRimages/255
HRimages = (HRimages-127.5)/127.5

psnr = metrics()
plot = plots(LRDim, HRDim, LRimages, HRimages, store_path)

def get_gan_network(generator, discriminator, optimizer):
    # discriminator.trainable = False
    # set_trainable(discriminator, False)

    gan_input = Input(shape=(LRDim, LRDim, 3))

    generated_images = generator(gan_input)

    gan_output = discriminator(generated_images)

    gan = Model(inputs=gan_input, outputs=[generated_images, gan_output])

    gan.compile(loss=['mse', 'binary_crossentropy'], optimizer=optimizer, metrics=[psnr.PSNR, 'accuracy'])
    return gan

def set_trainable(model, flag):
    for l in model.layers:
        l.trainable = flag

gan_opt = Adam(lr=0.00009, beta_1=0.5, beta_2=0.999, epsilon=1e-08)
d_opt = Adam(lr=0.00009, beta_1=0.5, beta_2=0.999, epsilon=1e-08)


if model_checkpoint == 'y':
    print("Loading the saved models: ")
    generator = load_model(model_path + "\\generator.hd5")
    discriminator = load_model(model_path + "\\discriminator.hd5")
    set_trainable(discriminator, True)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)
    gan = load_model(model_path + "\\gan.hd5", custom_objects={'PSNR': psnr.PSNR})
else:
    generator = generator(LRDim, num_residual_blocks)
    discriminator = discriminator(HRDim, d_opt)
    generator = generator.get_generator()
    discriminator = discriminator.get_discriminator()
    gan = get_gan_network(generator, discriminator, gan_opt)

discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)
gan.compile(loss=['mse', 'binary_crossentropy'], optimizer=gan_opt, metrics=[psnr.PSNR, 'accuracy'])


epochs = 50
batch_size = 8

batch_count = len(LRimages) // batch_size

loss = {'d_loss':[], 'g_loss':[]}

tb_dis = keras.callbacks.TensorBoard(
  log_dir=store_path + '\\dis_logs',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tb_dis.set_model(discriminator)

tb_gan = keras.callbacks.TensorBoard(
  log_dir=store_path + '\\gan_logs',
  histogram_freq=0,
  batch_size=batch_size,
  write_graph=True,
  write_grads=True
)
tb_gan.set_model(gan)


def training(start, steps):
    for _ in (range(start, steps)):
        rand_index = np.random.randint(0, len(LRimages), size=batch_size)
        image_batch_HR = HRimages[rand_index]
        image_batch_LR = LRimages[rand_index]
        generated_images = generator.predict(image_batch_LR)

        y_real = np.ones(batch_size) - np.random.random_sample(batch_size) * 0.2
        y_fake = np.random.random_sample(batch_size) * 0.2

        set_trainable(discriminator, True)

        real_loss = 0
        fake_loss = 0

        for __ in range(d_train):
            real_fit = discriminator.train_on_batch(image_batch_HR, y_real)
            tb_dis.on_batch_end(_, logs=real_fit)

            fake_fit = discriminator.train_on_batch(generated_images, y_fake)
            tb_dis.on_batch_end(_, logs=fake_fit)

            real_loss += real_fit
            fake_loss += fake_fit

        d_loss = np.add(real_loss/d_train, fake_loss/d_train)


        set_trainable(discriminator, False)

        g_loss = gan.train_on_batch(image_batch_LR, [image_batch_HR, y_real])

        tb_gan.on_batch_end(_, logs=g_loss)

        loss['g_loss'].append(g_loss[1])
        loss['d_loss'].append(d_loss)
        print("Step: ", _, " Discriminator Loss: ", d_loss, " Real Loss: ", real_loss/d_train, " Fake Loss: ", fake_loss/d_train, " Generator Loss: ", g_loss[1], " PSNR: ", g_loss[3])
        if _%500 == 0:
            plot.plot_generated_images(index=np.random.randint(0, len(image_batch_LR)), generator=generator, step=_)

        if (_ % 500) == 0 and _ != 0:
            generator.save(store_path + "\\generator.hd5")
            gan.save(store_path + "\\gan.hd5")
            discriminator.save(store_path + "\\discriminator.hd5")

d_train = 1
start = 0
steps = int(steps) + start

training(start, steps)

#
# generator.save(store_path + "\\generator.h5")
# gan.save(store_path + "\\gan.hd5")
# discriminator.save(store_path + "\\discriminator.hd5")
# len(loss['g_loss'])
# plt.plot(range(len(loss['g_loss'])), loss['g_loss'], color='b')
# plt.plot(range(len(loss['g_loss'])), loss['d_loss'], color='r')
# plt.show()

with open("keep_count.txt", 'w') as f:
    f.write(str(int(run_count) + 1))

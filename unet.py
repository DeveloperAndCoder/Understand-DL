from unet_model import *
import collect_data
import argparse
from pathlib import Path
import os
from keras.callbacks import CSVLogger, ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"]="2"

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--runnum", required=True,
	help="Run number: eg stl10_4")
args = vars(ap.parse_args())

print(args)

runnum = args["runnum"]

runnum.strip()

save_dir = "saved_models/" + runnum + "/"
log_dir = "Log/" + runnum + "/unet/"
checkpoint_dir = 'checkpoint/' + runnum + "/unet/"

Path(save_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)
Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

(x_train, y_train), (x_test, y_test) = collect_data.STL10.load_data(collect_data.STL10(), preprocess=True, toResize=True, dims=(96,96))
print(x_train.shape, x_train[0].shape)


print(type(x_train), type(x_test))


model = unet()
# exit(1)
csv_logger = CSVLogger(log_dir + "unet_log.csv", append=True, separator=';')
checkpoint_template = os.path.join(checkpoint_dir, "{epoch:03d}_{loss:.2f}.h5")
checkpoint = ModelCheckpoint(checkpoint_template, monitor='loss', save_weights_only=False, mode='auto', period=5, verbose=1)

model.fit(x=x_train, y=x_train, epochs=20, callbacks=[csv_logger, checkpoint])

model_path = os.path.join(save_dir, 'unet.h5')
model.save(model_path)
# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)

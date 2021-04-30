from utils_mouth import mkDir, Mouths, eGAN
# from eGAN_mouth import eGAN
import tensorflow as tf
import numpy as np, os
import os

os.environ['CUDA_VISIBLE_DEVICES']= '12'

flags = tf.app.flags
flags.DEFINE_string("path", '?', "path of training data")
FLAGS = flags.FLAGS

if __name__ == "__main__":

    logDir = "./Mouth_Impainting/outpout_mouth/log/logs{}".format(1)
    ckptDir = "./Mouth_Impainting/outpout_mouth/model_gan{}/".format("Experiment_4_21")
    sampleDir = "./Mouth_Impainting/outpout_mouth/sample{}/sample_{}".format(1, "Experiment_4_21")

    mkDir(logDir)
    mkDir(ckptDir)
    mkDir(sampleDir)

    data_ob = Mouths(FLAGS.path)

    eGAN = eGAN(batch_size= 4, max_iters= 100000,
                      model_path= ckptDir, data_ob= data_ob, sample_path= sampleDir , log_dir=logDir,
                      learning_rate= 0.0001, is_load=True, lam_recon=1, lam_gp=10,
                    use_sp=True, beta1=0.5, beta2=0.975, n_critic=1)

    eGAN.build_model_GAN()
    eGAN.train()




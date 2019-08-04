import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import spatial_transformer

class SpatialTransformerTest(tf.test.TestCase):
  def test_batch_transform(self):
    transformer = spatial_transformer.SpatialTransformer(
      output_image_size=(780, 1040),
      num_control_points=4,
      init_bias_pattern='identity',
      margins=(0,0),
      activation='sigmoid'
    )

    output_image_size=(780, 1040)
    image_ctrl_pts_1 = np.array([[[345, 3], [927, 387], [113, 456], [750, 687]]], dtype=np.int32)
    image_ctrl_pts_2 = np.array([[[345, 3], [927, 387], [113, 456], [750, 687]]], dtype=np.int32)
    image_ctrl_pts_3 = np.array([[[345, 3], [927, 387], [113, 456], [750, 687]]], dtype=np.int32)
    image_ctrl_pts_1 = image_ctrl_pts_1.astype(np.float32)
    array1 = image_ctrl_pts_1[:,:,0]
    array2 = image_ctrl_pts_1[:,:,1]
    array1 = array1/output_image_size[1]
    array2 = array2/output_image_size[0]
    image_ctrl_pts_1 = np.around(np.stack([array1,array2],axis=2),2)

    image_ctrl_pts_2 = image_ctrl_pts_2.astype(np.float32)
    array1 = image_ctrl_pts_2[:,:,0]
    array2 = image_ctrl_pts_2[:,:,1]
    array1 = array1/output_image_size[1]
    array2 = array2/output_image_size[0]
    image_ctrl_pts_2 = np.around(np.stack([array1,array2],axis=2),2)

    image_ctrl_pts_3 = image_ctrl_pts_3.astype(np.float32)
    array1 = image_ctrl_pts_3[:,:,0]
    array2 = image_ctrl_pts_3[:,:,1]
    array1 = array1/output_image_size[1]
    array2 = array2/output_image_size[0]
    image_ctrl_pts_3 = np.around(np.stack([array1,array2],axis=2),2)

    test_input_ctrl_pts = np.concatenate((image_ctrl_pts_1,image_ctrl_pts_2,image_ctrl_pts_3),axis=0)

    test_im = Image.open('/home/AI/chencong/Caster/data/313.jpg').convert('L')#.resize((32, 71))
    test_image_array = np.array(test_im)
    test_image_array = np.array([test_image_array, test_image_array, test_image_array])
    test_images = tf.cast(tf.constant(test_image_array), tf.float32)
    test_images = (test_images / 128.0) - 1.0

    sampling_grid = transformer._batch_generate_grid(test_input_ctrl_pts)
    print('11111111111111111111',sampling_grid)
    rectified_images = transformer._batch_sample(test_images, sampling_grid)
    rectified_images = tf.squeeze(rectified_images,[3])
    print('22222222222222222222',rectified_images)

    output_ctrl_pts = transformer._output_control_points
    with self.test_session() as sess:
      outputs = sess.run({
        'sampling_grid': sampling_grid,
        'rectified_images': rectified_images
      })

      rectified_images = outputs['rectified_images']
      rectified_images = (rectified_images + 1.0) * 128.0
      print(rectified_images[0])
      if True:
        plt.figure()
        plt.subplot(3,4,1)
        plt.scatter(test_input_ctrl_pts[0,:,0], test_input_ctrl_pts[0,:,1])
        plt.subplot(3,4,2)
        plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
        plt.subplot(3,4,3)
        plt.scatter(outputs['sampling_grid'][0,:,0], outputs['sampling_grid'][0,:,1], marker='+')
        plt.subplot(3,4,4)
        plt.imshow(rectified_images[0].astype(np.uint8))
        img = Image.fromarray(rectified_images[0].astype(np.uint8)).convert('RGB')
        img.save('/home/AI/chencong/Caster/data/rectified_image_1_1.jpg')

        plt.subplot(3,4,5)
        plt.scatter(test_input_ctrl_pts[1,:,0], test_input_ctrl_pts[1,:,1])
        plt.subplot(3,4,6)
        plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
        plt.subplot(3,4,7)
        plt.scatter(outputs['sampling_grid'][1,:,0], outputs['sampling_grid'][1,:,1], marker='+')
        plt.subplot(3,4,8)
        plt.imshow(rectified_images[1].astype(np.uint8))
        img = Image.fromarray(rectified_images[1].astype(np.uint8))
        img.save('/home/AI/chencong/Caster/data/rectified_image_1_2.jpg')

        plt.subplot(3,4,9)
        plt.scatter(test_input_ctrl_pts[2,:,0], test_input_ctrl_pts[2,:,1])
        plt.subplot(3,4,10)
        plt.scatter(output_ctrl_pts[:,0], output_ctrl_pts[:,1])
        plt.subplot(3,4,11)
        plt.scatter(outputs['sampling_grid'][2,:,0], outputs['sampling_grid'][2,:,1], marker='+')
        plt.subplot(3,4,12)
        plt.imshow(rectified_images[2].astype(np.uint8))
        img = Image.fromarray(rectified_images[2].astype(np.uint8))
        img.save('/home/AI/chencong/Caster/data/rectified_image_1_3.jpg')

        plt.show()

    
if __name__ == '__main__':
  tf.test.main()

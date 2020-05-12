from styx_msgs.msg import TrafficLight
import tensorflow as tf
import rospy
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        SSD_GRAPH_FILE = rospy.get_param("/light_classification_model_config")
        #rospy.logwarn(SSD_GRAPH_FILE)
        self.graph = tf.Graph()
        with self.graph.as_default():
	    od_graph_def = tf.GraphDef()
	    with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
	        serialized_graph = fid.read()
	        od_graph_def.ParseFromString(serialized_graph)
	        tf.import_graph_def(od_graph_def, name='')
	self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

	# Each box represents a part of the image where a particular object was detected.
	self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

	# Each score represent how level of confidence for each of the objects.
	# Score is shown on the result image, together with the class label.
	self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

	# The classification of the object (integer id).
	self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')

	self.tl_list = [TrafficLight.UNKNOWN, TrafficLight.GREEN, TrafficLight.YELLOW, TrafficLight.RED]
        self.count = 0;

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        self.count += 1
        draw = Image.fromarray(image[0])
        draw.save('/data/git/f/output/{}.jpg'.format(self.count))
        """

        #TODO implement light color prediction
        with tf.Session(graph=self.graph) as sess:
	    # Actual detection.
	    (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
		                                feed_dict={self.image_tensor: image})

	    # Remove unnecessary dimensions
	    boxes = np.squeeze(boxes)
	    scores = np.squeeze(scores)
	    classes = np.squeeze(classes)

	    weight = [0.0, 0.0, 0.0, 0.0]

            for i in range(len(classes)):
                class_id = int(classes[i])
                score = scores[i]
	        #rospy.logwarn('count: %s, class_id: %s, score: %s, x_width: %s, xmin: %s, y_height: %s, ymin: %s', self.count, class_id, score, (boxes[i][3] - boxes[i][1]) * 800,  boxes[i][1] * 800, (boxes[i][2] - boxes[i][0]) * 600, boxes[i][0] * 600)
                if score > 0.6:
		    if class_id == 1:
		        weight[1] += score
		    elif class_id == 2:
		        weight[3] += score
		    elif class_id == 3:
		        weight[2] += score
		    elif class_id == 4:
	    	        weight[0] += score

            max_weight = max(weight)
            if max_weight > 0.6:
	        best_index = weight.index(max_weight)
	        #rospy.logwarn(best_index)
	        return self.tl_list[best_index]

import os
import sys
import socket
import time
import tensorflow as tf
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
from cassandra.query import SimpleStatement
from flask import Flask, request
from PIL import Image, ImageFilter
from redis import Redis, RedisError
import logging

log = logging.getLogger()
log.setLevel('INFO')
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
log.addHandler(handler)

# Connect to Redis
redis = Redis(host="redis", db=0, socket_connect_timeout=2, socket_timeout=2)

app = Flask(__name__)


### Model setup
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
saver.restore(sess, "./model/model")


### Connect to the cassandra
KEYSPACE="mykeyspace_mnist"
def createKeySpace():
    cluster = Cluster(contact_points=['172.17.0.1'],port=9042)
    session = cluster.connect()

    log.info("Creating keyspace...")
    try:
        session.execute("""
            CREATE KEYSPACE %s
            WITH replication = { 'class': 'SimpleStrategy', 'replication_factor': '2' }
            """ % KEYSPACE)

        log.info("setting keyspace...")
        session.set_keyspace(KEYSPACE)

        log.info("creating table...")
        session.execute("""
            CREATE TABLE mytable (
                mykey text,
                col1 text,
                col2 int,
                PRIMARY KEY (mykey, col1)
            )
            """)
    except Exception as e:
        log.error("Unable to create keyspace")
        log.error(e)

# createKeySpace()


def deleteTable():
    cluster = Cluster(contact_points=['172.17.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    try:
        log.info("Deleting a table...")
        session.execute('''DROP TABLE mytable''')
    except Exception as e:
        log.error("Unable to delete a table")
        log.error(e)

'''use python to delete the created keyspace'''

def deleteKeyspace():
    cluster = Cluster(contact_points=['172.17.0.1'],port=9042)
    session = cluster.connect()

    try:
        log.info("Deleting a keyspace...")
        session.execute('''DROP KEYSPACE %s''' % KEYSPACE)
    except Exception as e:
        log.error("Unable to delete a keyspace")
        log.error(e)



'''use python to insert a few records in our table'''
# 插入识别图片的时间戳time、文件名name、识别结果value
def insertData(time, name, value):
    cluster = Cluster(contact_points=['172.17.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    prepared = session.prepare("""
    INSERT INTO mytable (mykey, col1, col2)
    VALUES (?, ?, ?)
    """)

    log.info("inserting into mytable")
    session.execute(prepared.bind((time, name, value)))
    # session.execute('''insert into mykeyspace.mytable(mykey,col1,col2) values(%s,%s,%d)''' %(time, name, value))

    # for i in range(number):
    #     if(i%5 == 0):
    #         log.info("inserting row %d" % i)
    #     session.execute(prepared.bind(("rec_key_%d" % i, 'aaa', 'bbb')))


'''Reading the freshly inserted data is not that difficult using a function similar to the one below:'''
def readRows():
    cluster = Cluster(contact_points=['172.17.0.1'],port=9042)
    session = cluster.connect()

    log.info("setting keyspace...")
    session.set_keyspace(KEYSPACE)

    rows = session.execute("SELECT * FROM mytable")
    log.info("key\tcol1\tcol2")
    log.info("---------\t----\t----")

    count=0
    for row in rows:
        if(count%100==0):
            log.info('\t'.join(row))
        count=count+1

    log.info("Total")
    log.info("-----")
    log.info("rows %d" %(count))

'''
def buildmodel()
    ## This is where the data gets imported
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Model Creation - Placeholders & Variables
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # Defining the loss and optimizations
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # The model is saved to disk as a model.ckpt file

    # Initializing the Session
    with tf.Session() as sess:
        sess.run(init)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
        save_path = saver.save(sess, "./model/")
    print ("Model saved in file: ", save_path)
'''

###
# @app.route("/prediction", methods=['GET','POST'])
def predictint():
    f = request.files["file"]   
    file_name = request.files["file"].filename
    imvalu = prepareImage(f)
    prediction = tf.argmax(y,1)
    pre = prediction.eval(feed_dict={x: [imvalu]}, session=sess)
    return str(pre[0])
    #return "The number upload is: [%s]" % str(pre[0])

def prepareImage(i):
    im = Image.open(i).convert('L')# 读图片并转为黑白的
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #将28X28矩阵改成一个784维向量
    
    if width != 28 or height !=28:
        im=im.resize((28,28),Image.ANTIALIAS)
    arr = []
    for j in range(28):
        for k in range(28):
            pixel = float(1.0-float(img.getpixel((j,k)))/255.0)
            arr.append(pixel) 
    return arr


'''
@app.route('/')
def index():
    try:
        visits = redis.incr("counter")
    except RedisError:
        visits = "<i>cannot connect to Redis, counter disabled</i>"

    html = '''
    <!doctype html>
    <html>
    <body>
    <form action='/prediction' method='post' enctype='multipart/form-data'>
        <input type='file' name='file'>
    <input type='submit' value='Upload'>
    </form>
    '''   
    return html.format(name=os.getenv("NAME", "MNIST"), hostname=socket.gethostname(), visits=visits) 
'''

if __name__ == "__main__":
    createKeySpace()
    # trainApp = TrainMnist(CKPT_DIR)
    # trainApp.train()
    # trainApp.calculate_accuracy()

    for i in range(10):
        a = predictint('test_images/%d.png' % i)
        insertData("2018.4.17_%d" % i, "%d.png" % i, a)

#   app.run(host='0.0.0.0',port=5000)
# expose to all, can be accessed by LAN users  

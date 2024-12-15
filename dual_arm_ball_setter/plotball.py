'''plotball.py

   Plot the /ball recorded in the ROS2 bag.
'''

import rclpy
import json
import numpy as np
import matplotlib.pyplot as plt

import glob, os, sys

from rosbag2_py                 import SequentialReader
from rosbag2_py._storage        import StorageOptions, ConverterOptions
from rclpy.serialization        import deserialize_message

from std_msgs.msg               import Float64
from std_msgs.msg               import String

#
#  Plot the Ball Data
#
def plotball(condmsgs, t0, bagname):
    # Process the ball messages.
    ball = np.array([list(json.loads(msg.data)) for msg in condmsgs])

    # Set up time, assuming a 100Hz rate.
    N  = len(ball)
    dt = 0.01
    t  = dt * np.linspace(0, N, N, endpoint=False)

    # Create a figure to plot ball vs. t
    fig, ax = plt.subplots(1, 1)

    # Plot the data.
    ax.plot(t, ball[:, 0])
    ax.plot(t, ball[:, 1])
    ax.plot(t, ball[:, 2])
    ax.legend(['x component', 'y component', 'z component'])
    ax.set(ylabel='Velocity of Ball Components')
    ax.set(xlabel='Time (sec)')
    ax.grid()


#
#  Main Code
#
def main():
    # Grab the arguments.
    bagname   = 'latest' if len(sys.argv) < 2 else sys.argv[1]

    # Check for the latest ROS bag:
    if bagname == 'latest':
        # Report.
        print("Looking for latest ROS bag...")

        # Look at all bags, making sure we have at least one!
        dbfiles = glob.glob('*/*.db3')
        if not dbfiles:
            raise FileNoFoundError('Unable to find a ROS2 bag')

        # Grab the modification times and the index of the newest.
        dbtimes = [os.path.getmtime(dbfile) for dbfile in dbfiles]
        i = dbtimes.index(max(dbtimes))

        # Select the newest.
        bagname = os.path.dirname(dbfiles[i])

    # Report.
    print("Reading ROS bag '%s'"  % bagname)


    # Set up the BAG reader.
    reader = SequentialReader()
    try:
        reader.open(StorageOptions(uri=bagname, storage_id='sqlite3'),
                    ConverterOptions('', ''))
    except Exception as e:
        print("Unable to read the ROS bag '%s'!" % bagname)
        print("Does it exist and WAS THE RECORDING Ctrl-c KILLED?")
        raise OSError("Error reading bag - did recording end?") from None

    # Get the starting time.
    t0 = reader.get_metadata().starting_time.nanoseconds * 1e-9 - 0.01

    # Get the topics and types:
    print("The bag contain message for:")
    for x in reader.get_all_topics_and_types():
        print("  topic %-20s of type %s" % (x.name, x.type))


    # Pull out the relevant messages.
    condmsgs = []
    while reader.has_next():
        # Grab a message.
        (topic, rawdata, timestamp) = reader.read_next()

        # Pull out the deserialized message.
        if   topic == '/ball':
            condmsgs.append(deserialize_message(rawdata, String))


    # Process the ball.
    if condmsgs:
        print("Plotting ball data...")
        plotball(condmsgs, t0, bagname)
    else:
        raise ValueError("No ball data!")

    # Show
    plt.show()


#
#   Run the main code.
#
if __name__ == "__main__":
    main()

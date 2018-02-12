import cv2
import argparse
import vpl
from networktables import NetworkTables
import socket
import capture

capture.parser = argparse.ArgumentParser(description='Example webcam view for punkvision')

capture.parser.add_argument('--source', type=str, default="0", help='camera number, image glob (like "data/*.png"), or video file ("x.mp4")')
capture.parser.add_argument('--size', type=int, nargs=2, default=(640, 480), help='image size')
capture.parser.add_argument('--blur', type=int, nargs=2, default=(12,12), help='blur size')

capture.parser.add_argument('--save', default=None, help='save the stream to files (ex: "data/{num}.png"')

capture.parser.add_argument('--stream', default=None, type=int, help='stream to a certain port (ex: "5802" and then connect to "localhost:5802" to view the stream)')

capture.parser.add_argument('--noshow', action='store_true', help="if this flag is set, do not show a window (useful for raspberry PIs without a screen, you can use --stream)")

capture.parser.add_argument('--noprop', action='store_true', help="if this flag is set, do not set capture properties (some may fail without this, use this if you are getting a QBUF error)")

args = capture.parser.parse_args()


pipe = vpl.Pipeline("record")

cam_props = vpl.CameraProperties()

args = capture.parser.parse_args()

# set preferred width and height
if not args.noprop:
    cam_props["FRAME_WIDTH"] = args.size[0]
    cam_props["FRAME_HEIGHT"] = args.size[1]

# find the source
pipe.add_vpl(vpl.VideoSource(source=args.source, properties=cam_props))

# resize
pipe.add_vpl(vpl.Resize(w=args.size[0], h=args.size[1]))









# add a FPS counter
pipe.add_vpl(vpl.FPSCounter())

#kill program
pipe.add_vpl(vpl.KillSwitch())


#stream it
if not args.noshow:
    pipe.add_vpl(vpl.Display(title="footage from " + str(args.source)))
if args.stream is not None:
    pipe.add_vpl(vpl.MJPGServer(port=args.stream))

    #table.putString("PiCamera", "/CameraPublisher/PiCamera/streams=['mjpeg:http://10.39.66.73:5802/?action=stream'")


try:
      # we let our VideoSource do the processing, autolooping
      pipe.process(image=None, data=None, loop=True)
except (KeyboardInterrupt, SystemExit):
    print("keyboard interrupt, quitting")
    exit(0)
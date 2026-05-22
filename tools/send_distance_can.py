#!/usr/bin/env python3
"""Send a distance value on SocketCAN using python-can.

Usage:
  python tools/send_distance_can.py --channel can0 --id 0x401 --distance 1234

Sends distance as unsigned 16-bit millimeters in little-endian in data bytes.
"""
import argparse
import struct
import sys

def main():
    parser = argparse.ArgumentParser(description="Send distance over SocketCAN")
    parser.add_argument("--channel", default="can0", help="SocketCAN channel (default: can0)")
    parser.add_argument("--id", default="401", help="CAN arbitration id (hex), e.g. 0x401")
    parser.add_argument("--distance", type=float, help="Distance in millimeters (or meters if --meters) ")
    parser.add_argument("--bus-type", default="socketcan", help="python-can bus type (default: socketcan)")
    args = parser.parse_args()

    if args.distance is None:
        print("Provide a distance with --distance", file=sys.stderr)
        sys.exit(2)

    dist_mm = int(args.distance)
    # Ensure the value fits in 2 bytes (unsigned 16-bit)
    if dist_mm < 0 or dist_mm > 0xFFFF:
        print("Distance out of range for 2 bytes (0..65535 mm)", file=sys.stderr)
        sys.exit(2)
    data = struct.pack("<H", dist_mm)  # 2 bytes little-endian (unsigned 16-bit)
    arb_id = int(args.id, 0)

    try:
        import can
    except Exception as e:
        print("python-can is required. Install with: pip install python-can", file=sys.stderr)
        raise

    bus = can.interface.Bus(channel=args.channel, bustype=args.bus_type)
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=False)
    try:
        bus.send(msg)
        print(f"Sent distance={dist_mm} mm on {args.channel} id=0x{arb_id:x}")
    except can.CanError as e:
        print("Failed to send CAN message:", e, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

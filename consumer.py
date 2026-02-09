"""
Huey consumer configuration file.
This file is required by huey's consumer command.
"""
# import both huey instance and task to ensure registration
from transcriber_huey import huey, transcribe_payload_task


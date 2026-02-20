"""
Huey consumer configuration file.
This file is required by huey's consumer command.
"""
# import both huey instance and task to ensure registration
from transcriber_huey import huey, transcribe_payload_task
# also import from transcriber_huey_queue to register tasks queued by queue_simulator
# explicitly import the task to ensure it's registered
from transcriber_huey_queue import transcribe_payload_task as queue_transcribe_payload_task


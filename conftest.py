import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# RLlib imports TensorFlow in every process if it is installed, even though this
# project uses torch only. That costs tens of seconds per pytest run and a lot of
# RAM. This switch (read when `ray.rllib` is first imported, and inherited by Ray
# worker processes) makes RLlib skip it.
os.environ.setdefault("RLLIB_TEST_NO_TF_IMPORT", "1")
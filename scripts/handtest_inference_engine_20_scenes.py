from __future__ import annotations

import sys

from handtest_inference_engine_real import main


if __name__ == "__main__":
    if "--max-scenes" not in sys.argv:
        sys.argv.extend(["--max-scenes", "20"])
    if "--experiment-id" not in sys.argv:
        sys.argv.extend(["--experiment-id", "ie_real_20"])
    main()

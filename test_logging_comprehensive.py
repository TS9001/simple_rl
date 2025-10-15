#!/usr/bin/env python3
"""
Test comprehensive logging to verify ALL output is captured.

This test verifies that:
1. Logger messages (info, warning, error) are captured
2. Print statements (stdout) are captured
3. Error messages (stderr) are captured
4. Exceptions with tracebacks are captured
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from simple_rl.utils.logging import setup_logging


def test_logging():
    """Test that all types of output are captured to log file."""

    print("\n" + "="*60)
    print("TESTING COMPREHENSIVE LOGGING")
    print("="*60 + "\n")

    # Setup logging with file output
    log_file = "test_logging_output.log"
    logger = setup_logging(level="INFO", log_file=log_file)

    print("="*60)
    print("TEST 1: Logger messages")
    print("="*60)
    logger.info("This is an INFO message from logger")
    logger.warning("This is a WARNING message from logger")
    logger.error("This is an ERROR message from logger")

    print("\n" + "="*60)
    print("TEST 2: Print statements (stdout)")
    print("="*60)
    print("This is a print() statement - should appear in log file!")
    print("Multiple lines:")
    print("  - Line 1")
    print("  - Line 2")
    print("  - Line 3")

    print("\n" + "="*60)
    print("TEST 3: Error messages (stderr)")
    print("="*60)
    print("Writing to stderr...", file=sys.stderr)
    print("Error message: Something went wrong!", file=sys.stderr)

    print("\n" + "="*60)
    print("TEST 4: Exception handling")
    print("="*60)
    try:
        # This will raise an exception
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.error(f"Caught exception: {e}")
        logger.exception("Full traceback:")

    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"\n✓ All output should be in: {log_file}")
    print("✓ Check the file to verify ALL messages were captured")
    print("✓ File should contain:")
    print("  - Logger messages (info, warning, error)")
    print("  - Print statements")
    print("  - Stderr messages")
    print("  - Exception tracebacks")

    # Read and display log file contents
    print("\n" + "="*60)
    print("LOG FILE CONTENTS:")
    print("="*60)
    with open(log_file, 'r') as f:
        contents = f.read()
        print(contents)

    # Verify all expected content is in the log
    expected_strings = [
        "INFO message from logger",
        "WARNING message from logger",
        "ERROR message from logger",
        "This is a print() statement",
        "Writing to stderr",
        "Error message: Something went wrong!",
        "Caught exception",
        "ZeroDivisionError",
        "Full traceback",
    ]

    print("\n" + "="*60)
    print("VERIFICATION:")
    print("="*60)
    all_found = True
    for expected in expected_strings:
        if expected in contents:
            print(f"✓ Found: {expected}")
        else:
            print(f"✗ MISSING: {expected}")
            all_found = False

    if all_found:
        print("\n🎉 SUCCESS! All output was captured to log file!")
    else:
        print("\n❌ FAILURE! Some output was NOT captured!")
        sys.exit(1)


if __name__ == "__main__":
    test_logging()

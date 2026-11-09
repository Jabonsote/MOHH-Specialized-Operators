
import sys

def clear_above_lines(lines: int = 2) -> None:
    """
    Clear n lines above the current line.
    """
    for _ in range(lines):
        sys.stdout.write('\033[F\033[K') 

def update_lines(lines: list[str]) -> None:
    """
    Update the len(lines) upper lines in the terminal.
    """

    clear_above_lines(len(lines))
    
    for line in lines:
        print(line)
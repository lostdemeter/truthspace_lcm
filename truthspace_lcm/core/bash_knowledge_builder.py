"""
Bash Knowledge Builder

Builds a comprehensive knowledge base for Bash shell commands,
including file operations, text processing, and system administration.

This populates the KnowledgeManager with detailed entries
that enable shell script generation from natural language.
"""

import os
import sys
from typing import Dict, List, Any

from truthspace_lcm.core.knowledge_manager import KnowledgeManager, KnowledgeDomain, KnowledgeEntry


class BashKnowledgeBuilder:
    """Builds Bash shell command knowledge base."""
    
    def __init__(self, manager: KnowledgeManager):
        self.manager = manager
    
    def build_all(self) -> int:
        """Build all Bash knowledge. Returns count of entries created."""
        count = 0
        count += self._build_file_operations()
        count += self._build_directory_operations()
        count += self._build_text_processing()
        count += self._build_file_viewing()
        count += self._build_permissions()
        count += self._build_system_info()
        count += self._build_process_management()
        count += self._build_networking()
        count += self._build_compression()
        count += self._build_common_patterns()
        return count
    
    def _create(self, name: str, entry_type: str, description: str,
                keywords: List[str], metadata: Dict[str, Any] = None) -> KnowledgeEntry:
        """Helper to create programming knowledge entry."""
        return self.manager.create(
            name=name,
            domain=KnowledgeDomain.PROGRAMMING,
            entry_type=entry_type,
            description=description,
            keywords=["bash", "shell", "command"] + keywords,
            metadata=metadata or {}
        )
    
    def _build_file_operations(self) -> int:
        """Build knowledge about file operations."""
        entries = [
            ("touch", "command",
             "Create empty file or update timestamp of existing file",
             ["create", "file", "empty", "new", "touch", "timestamp"],
             {"syntax": "touch <filename>",
              "example": "touch newfile.txt",
              "flags": "-a (access time), -m (modification time)"}),
            
            ("cp", "command",
             "Copy files or directories",
             ["copy", "duplicate", "file", "cp"],
             {"syntax": "cp <source> <destination>",
              "example": "cp file.txt backup.txt",
              "flags": "-r (recursive), -i (interactive), -v (verbose)"}),
            
            ("mv", "command",
             "Move or rename files and directories",
             ["move", "rename", "file", "mv"],
             {"syntax": "mv <source> <destination>",
              "example": "mv oldname.txt newname.txt",
              "flags": "-i (interactive), -v (verbose), -n (no overwrite)"}),
            
            ("rm", "command",
             "Remove/delete files or directories",
             ["remove", "delete", "file", "rm", "erase"],
             {"syntax": "rm <filename>",
              "example": "rm unwanted.txt",
              "flags": "-r (recursive), -f (force), -i (interactive)",
              "warning": "Be careful with rm -rf!"}),
            
            ("ln", "command",
             "Create links between files (symbolic or hard)",
             ["link", "symlink", "symbolic", "ln"],
             {"syntax": "ln -s <target> <linkname>",
              "example": "ln -s /path/to/file linkname",
              "flags": "-s (symbolic link), -f (force)"}),
            
            ("file", "command",
             "Determine file type",
             ["type", "file", "identify", "format"],
             {"syntax": "file <filename>",
              "example": "file document.pdf",
              "returns": "File type description"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_directory_operations(self) -> int:
        """Build knowledge about directory operations."""
        entries = [
            ("mkdir", "command",
             "Create new directory/folder",
             ["create", "directory", "folder", "mkdir", "make"],
             {"syntax": "mkdir <dirname>",
              "example": "mkdir new_folder",
              "flags": "-p (create parents), -v (verbose)"}),
            
            ("rmdir", "command",
             "Remove empty directory",
             ["remove", "delete", "directory", "folder", "rmdir"],
             {"syntax": "rmdir <dirname>",
              "example": "rmdir empty_folder",
              "note": "Only works on empty directories"}),
            
            ("cd", "command",
             "Change current directory",
             ["change", "directory", "navigate", "cd", "go"],
             {"syntax": "cd <path>",
              "example": "cd /home/user/documents",
              "shortcuts": "cd ~ (home), cd - (previous), cd .. (parent)"}),
            
            ("pwd", "command",
             "Print current working directory",
             ["print", "current", "directory", "pwd", "where"],
             {"syntax": "pwd",
              "example": "pwd",
              "returns": "Absolute path of current directory"}),
            
            ("ls", "command",
             "List directory contents",
             ["list", "directory", "files", "ls", "show", "contents"],
             {"syntax": "ls [options] [path]",
              "example": "ls -la",
              "flags": "-l (long), -a (all), -h (human), -R (recursive)"}),
            
            ("tree", "command",
             "Display directory structure as tree",
             ["tree", "structure", "hierarchy", "display"],
             {"syntax": "tree [path]",
              "example": "tree -L 2",
              "flags": "-L <level> (depth), -d (dirs only)"}),
            
            ("find", "command",
             "Search for files in directory hierarchy",
             ["find", "search", "locate", "files"],
             {"syntax": "find <path> [options]",
              "example": "find . -name '*.txt'",
              "flags": "-name (pattern), -type (f/d), -mtime (modified)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_text_processing(self) -> int:
        """Build knowledge about text processing commands."""
        entries = [
            ("grep", "command",
             "Search for patterns in text/files",
             ["search", "pattern", "find", "text", "grep", "match"],
             {"syntax": "grep <pattern> <file>",
              "example": "grep 'error' logfile.txt",
              "flags": "-i (ignore case), -r (recursive), -n (line numbers), -v (invert)"}),
            
            ("sed", "command",
             "Stream editor for text transformation",
             ["replace", "substitute", "edit", "text", "sed", "transform"],
             {"syntax": "sed 's/old/new/g' <file>",
              "example": "sed 's/foo/bar/g' file.txt",
              "flags": "-i (in-place), -e (expression)"}),
            
            ("awk", "command",
             "Pattern scanning and text processing",
             ["process", "columns", "fields", "text", "awk"],
             {"syntax": "awk '{print $1}' <file>",
              "example": "awk -F',' '{print $2}' data.csv",
              "flags": "-F (field separator)"}),
            
            ("sort", "command",
             "Sort lines of text",
             ["sort", "order", "arrange", "lines"],
             {"syntax": "sort <file>",
              "example": "sort -n numbers.txt",
              "flags": "-n (numeric), -r (reverse), -u (unique), -k (key)"}),
            
            ("uniq", "command",
             "Report or filter repeated lines",
             ["unique", "duplicate", "filter", "lines", "uniq"],
             {"syntax": "uniq <file>",
              "example": "sort file.txt | uniq -c",
              "flags": "-c (count), -d (duplicates only), -u (unique only)"}),
            
            ("wc", "command",
             "Count lines, words, characters in file",
             ["count", "lines", "words", "characters", "wc"],
             {"syntax": "wc <file>",
              "example": "wc -l file.txt",
              "flags": "-l (lines), -w (words), -c (bytes), -m (chars)"}),
            
            ("cut", "command",
             "Extract columns/fields from text",
             ["cut", "extract", "columns", "fields"],
             {"syntax": "cut -d',' -f1 <file>",
              "example": "cut -d':' -f1 /etc/passwd",
              "flags": "-d (delimiter), -f (fields), -c (characters)"}),
            
            ("tr", "command",
             "Translate or delete characters",
             ["translate", "replace", "characters", "tr", "convert"],
             {"syntax": "tr <set1> <set2>",
              "example": "echo 'hello' | tr 'a-z' 'A-Z'",
              "flags": "-d (delete), -s (squeeze)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_file_viewing(self) -> int:
        """Build knowledge about file viewing commands."""
        entries = [
            ("cat", "command",
             "Display file contents (concatenate)",
             ["display", "show", "view", "file", "cat", "print", "contents"],
             {"syntax": "cat <file>",
              "example": "cat file.txt",
              "flags": "-n (line numbers), -b (non-blank numbers)"}),
            
            ("less", "command",
             "View file with pagination (scrollable)",
             ["view", "page", "scroll", "less", "read"],
             {"syntax": "less <file>",
              "example": "less largefile.txt",
              "keys": "q (quit), / (search), n (next), g (top), G (bottom)"}),
            
            ("head", "command",
             "Display first lines of file",
             ["head", "first", "top", "beginning", "lines"],
             {"syntax": "head -n <count> <file>",
              "example": "head -n 10 file.txt",
              "flags": "-n (number of lines), -c (bytes)"}),
            
            ("tail", "command",
             "Display last lines of file",
             ["tail", "last", "end", "bottom", "lines"],
             {"syntax": "tail -n <count> <file>",
              "example": "tail -f logfile.txt",
              "flags": "-n (lines), -f (follow/live), -c (bytes)"}),
            
            ("diff", "command",
             "Compare two files line by line",
             ["compare", "difference", "diff", "files"],
             {"syntax": "diff <file1> <file2>",
              "example": "diff old.txt new.txt",
              "flags": "-u (unified), -y (side-by-side), -q (brief)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_permissions(self) -> int:
        """Build knowledge about permission commands."""
        entries = [
            ("chmod", "command",
             "Change file permissions (read/write/execute)",
             ["permissions", "chmod", "access", "rights", "executable"],
             {"syntax": "chmod <mode> <file>",
              "example": "chmod 755 script.sh",
              "modes": "755 (rwxr-xr-x), 644 (rw-r--r--), +x (add execute)"}),
            
            ("chown", "command",
             "Change file owner and group",
             ["owner", "chown", "ownership", "user", "group"],
             {"syntax": "chown <user>:<group> <file>",
              "example": "chown user:group file.txt",
              "flags": "-R (recursive)"}),
            
            ("chgrp", "command",
             "Change file group ownership",
             ["group", "chgrp", "ownership"],
             {"syntax": "chgrp <group> <file>",
              "example": "chgrp developers project/",
              "flags": "-R (recursive)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_system_info(self) -> int:
        """Build knowledge about system information commands."""
        entries = [
            ("echo", "command",
             "Print text to terminal/stdout",
             ["print", "echo", "output", "display", "text"],
             {"syntax": "echo <text>",
              "example": "echo 'Hello World'",
              "flags": "-n (no newline), -e (escape sequences)"}),
            
            ("whoami", "command",
             "Print current username",
             ["user", "username", "whoami", "current"],
             {"syntax": "whoami",
              "example": "whoami",
              "returns": "Current username"}),
            
            ("hostname", "command",
             "Print or set system hostname",
             ["hostname", "computer", "name", "system"],
             {"syntax": "hostname",
              "example": "hostname",
              "returns": "System hostname"}),
            
            ("uname", "command",
             "Print system information",
             ["system", "info", "uname", "kernel", "os"],
             {"syntax": "uname -a",
              "example": "uname -a",
              "flags": "-a (all), -r (kernel), -m (machine)"}),
            
            ("df", "command",
             "Display disk space usage",
             ["disk", "space", "usage", "df", "storage"],
             {"syntax": "df -h",
              "example": "df -h",
              "flags": "-h (human readable), -T (filesystem type)"}),
            
            ("du", "command",
             "Estimate file/directory space usage",
             ["disk", "usage", "size", "du", "space"],
             {"syntax": "du -sh <path>",
              "example": "du -sh /home/user",
              "flags": "-s (summary), -h (human), -a (all files)"}),
            
            ("date", "command",
             "Display or set system date/time",
             ["date", "time", "datetime", "timestamp"],
             {"syntax": "date [+format]",
              "example": "date '+%Y-%m-%d %H:%M:%S'",
              "formats": "%Y (year), %m (month), %d (day), %H:%M:%S (time)"}),
            
            ("env", "command",
             "Display environment variables",
             ["environment", "variables", "env", "config"],
             {"syntax": "env",
              "example": "env | grep PATH",
              "related": "export, printenv"}),
            
            ("which", "command",
             "Locate command executable",
             ["which", "locate", "find", "command", "path"],
             {"syntax": "which <command>",
              "example": "which python",
              "returns": "Path to executable"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_process_management(self) -> int:
        """Build knowledge about process management."""
        entries = [
            ("ps", "command",
             "Display running processes",
             ["process", "ps", "running", "list"],
             {"syntax": "ps aux",
              "example": "ps aux | grep python",
              "flags": "a (all users), u (user format), x (no tty)"}),
            
            ("top", "command",
             "Display real-time process information",
             ["top", "processes", "monitor", "cpu", "memory"],
             {"syntax": "top",
              "example": "top",
              "keys": "q (quit), k (kill), M (sort memory)"}),
            
            ("kill", "command",
             "Terminate process by PID",
             ["kill", "terminate", "stop", "process", "end"],
             {"syntax": "kill <pid>",
              "example": "kill -9 1234",
              "signals": "-9 (SIGKILL), -15 (SIGTERM), -HUP (reload)"}),
            
            ("killall", "command",
             "Kill processes by name",
             ["kill", "terminate", "name", "process"],
             {"syntax": "killall <name>",
              "example": "killall firefox",
              "flags": "-9 (force)"}),
            
            ("bg", "command",
             "Resume job in background",
             ["background", "bg", "job", "resume"],
             {"syntax": "bg [job]",
              "example": "bg %1"}),
            
            ("fg", "command",
             "Bring job to foreground",
             ["foreground", "fg", "job", "resume"],
             {"syntax": "fg [job]",
              "example": "fg %1"}),
            
            ("nohup", "command",
             "Run command immune to hangups",
             ["nohup", "background", "persistent", "detach"],
             {"syntax": "nohup <command> &",
              "example": "nohup ./script.sh &",
              "note": "Output goes to nohup.out"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_networking(self) -> int:
        """Build knowledge about networking commands."""
        entries = [
            ("curl", "command",
             "Transfer data from/to URL",
             ["curl", "http", "download", "url", "request", "api"],
             {"syntax": "curl <url>",
              "example": "curl -O https://example.com/file.txt",
              "flags": "-O (save), -o (output), -X (method), -d (data), -H (header)"}),
            
            ("wget", "command",
             "Download files from web",
             ["wget", "download", "http", "file", "web"],
             {"syntax": "wget <url>",
              "example": "wget https://example.com/file.zip",
              "flags": "-O (output), -c (continue), -r (recursive)"}),
            
            ("ping", "command",
             "Test network connectivity",
             ["ping", "network", "test", "connectivity"],
             {"syntax": "ping <host>",
              "example": "ping -c 4 google.com",
              "flags": "-c (count), -i (interval)"}),
            
            ("ssh", "command",
             "Secure shell remote login",
             ["ssh", "remote", "login", "secure", "shell"],
             {"syntax": "ssh user@host",
              "example": "ssh user@192.168.1.100",
              "flags": "-p (port), -i (identity file)"}),
            
            ("scp", "command",
             "Secure copy files over SSH",
             ["scp", "copy", "remote", "secure", "transfer"],
             {"syntax": "scp <source> <dest>",
              "example": "scp file.txt user@host:/path/",
              "flags": "-r (recursive), -P (port)"}),
            
            ("netstat", "command",
             "Network statistics and connections",
             ["netstat", "network", "connections", "ports"],
             {"syntax": "netstat -tulpn",
              "example": "netstat -tulpn | grep :80",
              "flags": "-t (tcp), -u (udp), -l (listening), -p (program)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_compression(self) -> int:
        """Build knowledge about compression commands."""
        entries = [
            ("tar", "command",
             "Archive files (tape archive)",
             ["tar", "archive", "compress", "extract", "pack"],
             {"syntax": "tar -czvf archive.tar.gz <files>",
              "example": "tar -xzvf archive.tar.gz",
              "flags": "-c (create), -x (extract), -z (gzip), -v (verbose), -f (file)"}),
            
            ("gzip", "command",
             "Compress files with gzip",
             ["gzip", "compress", "zip"],
             {"syntax": "gzip <file>",
              "example": "gzip file.txt",
              "flags": "-d (decompress), -k (keep original)"}),
            
            ("gunzip", "command",
             "Decompress gzip files",
             ["gunzip", "decompress", "unzip", "extract"],
             {"syntax": "gunzip <file.gz>",
              "example": "gunzip file.txt.gz"}),
            
            ("zip", "command",
             "Create zip archives",
             ["zip", "compress", "archive"],
             {"syntax": "zip archive.zip <files>",
              "example": "zip -r archive.zip folder/",
              "flags": "-r (recursive)"}),
            
            ("unzip", "command",
             "Extract zip archives",
             ["unzip", "extract", "decompress"],
             {"syntax": "unzip <archive.zip>",
              "example": "unzip archive.zip -d /path/",
              "flags": "-d (destination), -l (list)"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)
    
    def _build_common_patterns(self) -> int:
        """Build knowledge about common bash patterns."""
        entries = [
            ("create_directory_pattern", "pattern",
             "Create directory with parents if needed",
             ["create", "directory", "folder", "mkdir"],
             {"code": "mkdir -p path/to/directory"}),
            
            ("create_file_with_content_pattern", "pattern",
             "Create file with content using echo",
             ["create", "file", "content", "write"],
             {"code": "echo 'content here' > filename.txt"}),
            
            ("append_to_file_pattern", "pattern",
             "Append content to existing file",
             ["append", "add", "file", "content"],
             {"code": "echo 'new content' >> filename.txt"}),
            
            ("find_and_delete_pattern", "pattern",
             "Find and delete files matching pattern",
             ["find", "delete", "remove", "pattern"],
             {"code": "find . -name '*.tmp' -delete"}),
            
            ("search_in_files_pattern", "pattern",
             "Search for text in multiple files",
             ["search", "find", "text", "grep", "files"],
             {"code": "grep -r 'search_term' /path/to/search/"}),
            
            ("replace_in_file_pattern", "pattern",
             "Replace text in file using sed",
             ["replace", "substitute", "text", "sed"],
             {"code": "sed -i 's/old_text/new_text/g' filename.txt"}),
            
            ("backup_file_pattern", "pattern",
             "Create timestamped backup of file",
             ["backup", "copy", "timestamp", "file"],
             {"code": "cp file.txt file.txt.$(date +%Y%m%d_%H%M%S).bak"}),
            
            ("list_large_files_pattern", "pattern",
             "Find largest files in directory",
             ["large", "files", "size", "biggest"],
             {"code": "du -ah . | sort -rh | head -20"}),
            
            ("count_files_pattern", "pattern",
             "Count files in directory",
             ["count", "files", "number", "directory"],
             {"code": "find . -type f | wc -l"}),
            
            ("download_file_pattern", "pattern",
             "Download file from URL",
             ["download", "url", "file", "wget", "curl"],
             {"code": "curl -O https://example.com/file.txt"}),
            
            ("make_executable_pattern", "pattern",
             "Make script executable",
             ["executable", "script", "chmod", "run"],
             {"code": "chmod +x script.sh"}),
            
            ("watch_log_pattern", "pattern",
             "Watch log file in real-time",
             ["watch", "log", "tail", "follow", "live"],
             {"code": "tail -f /var/log/syslog"}),
            
            ("pipe_pattern", "pattern",
             "Pipe output between commands",
             ["pipe", "chain", "output", "input"],
             {"code": "command1 | command2 | command3"}),
            
            ("redirect_output_pattern", "pattern",
             "Redirect stdout and stderr to file",
             ["redirect", "output", "file", "log"],
             {"code": "command > output.log 2>&1"}),
            
            ("loop_files_pattern", "pattern",
             "Loop through files in directory",
             ["loop", "files", "iterate", "for"],
             {"code": "for file in *.txt; do echo \"$file\"; done"}),
            
            ("conditional_pattern", "pattern",
             "Conditional execution with if",
             ["if", "condition", "check", "test"],
             {"code": "if [ -f file.txt ]; then echo 'exists'; fi"}),
            
            ("check_file_exists_pattern", "pattern",
             "Check if file exists",
             ["check", "exists", "file", "test"],
             {"code": "[ -f filename ] && echo 'File exists'"}),
            
            ("check_dir_exists_pattern", "pattern",
             "Check if directory exists",
             ["check", "exists", "directory", "test"],
             {"code": "[ -d dirname ] && echo 'Directory exists'"}),
        ]
        
        for name, etype, desc, kw, meta in entries:
            self._create(name, etype, desc, kw, meta)
        return len(entries)


def build_bash_knowledge(storage_dir: str = None) -> KnowledgeManager:
    """Build and return a knowledge manager with Bash knowledge."""
    manager = KnowledgeManager(storage_dir=storage_dir)
    builder = BashKnowledgeBuilder(manager)
    count = builder.build_all()
    print(f"Built {count} Bash knowledge entries")
    return manager


if __name__ == "__main__":
    # Build knowledge in the project's knowledge_store directory
    storage_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "knowledge_store"
    )
    
    manager = build_bash_knowledge(storage_dir)
    
    print(f"\nKnowledge by domain: {manager.count_by_domain()}")
    print(f"\nStored in: {storage_dir}")
    
    # Test some queries
    print("\n" + "=" * 60)
    print("TESTING BASH QUERIES")
    print("=" * 60)
    
    test_queries = [
        ["create", "directory", "folder"],
        ["copy", "file"],
        ["search", "text", "grep"],
        ["download", "url", "file"],
        ["delete", "remove", "file"],
        ["list", "files", "directory"],
    ]
    
    for keywords in test_queries:
        print(f"\nQuery: {keywords}")
        results = manager.query(keywords, domain=KnowledgeDomain.PROGRAMMING, top_k=3)
        for sim, entry in results:
            if "bash" in entry.keywords or "shell" in entry.keywords:
                print(f"  → {entry.name}: {sim:.3f}")
                if entry.metadata.get("code"):
                    print(f"      Code: {entry.metadata['code']}")
                elif entry.metadata.get("example"):
                    print(f"      Example: {entry.metadata['example']}")

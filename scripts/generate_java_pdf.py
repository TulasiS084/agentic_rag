"""
Generate a rich, comprehensive Java_Collections_Framework_Handbook.pdf
covering overview, ArrayList, HashMap, Set/Queue, and time complexities.
"""
from pathlib import Path
import pymupdf

def create_handbook():
    pdf_path = Path(__file__).parent.parent / "sample_docs" / "Java_Collections_Framework_Handbook.pdf"
    pdf_path.parent.mkdir(exist_ok=True)

    doc = pymupdf.open()

    # --- Page 1: Overview & Framework Hierarchy ---
    p1 = doc.new_page()
    p1.insert_text(
        (40, 50),
        "Java Collections Framework Handbook (Master Notes)\n"
        "====================================================\n\n"
        "What this handbook is all about:\n"
        "This PDF is a Java Collections Framework handbook designed mainly for learning\n"
        "and interview preparation. It covers List, Set, Queue, Deque, Map, their\n"
        "implementations, internal working, commonly used methods, time complexities,\n"
        "and practical applications such as BFS and DFS.\n\n"
        "Core Framework Hierarchy:\n"
        "1. Iterable Interface -> Collection Interface\n"
        "   - List Interface: Ordered collection allowing duplicates (ArrayList, LinkedList, Vector, Stack).\n"
        "   - Set Interface: Unordered collection prohibiting duplicates (HashSet, LinkedHashSet, TreeSet).\n"
        "   - Queue & Deque Interface: FIFO / double-ended queues for graph algorithms (PriorityQueue, ArrayDeque, LinkedList).\n"
        "2. Map Interface (Separate Hierarchy): Stores key-value mappings (HashMap, LinkedHashMap, TreeMap, ConcurrentHashMap).\n\n"
        "Practical Graph & Interview Applications:\n"
        "- Breadth-First Search (BFS): Uses Queue / ArrayDeque for level-order traversal.\n"
        "- Depth-First Search (DFS): Uses Deque / ArrayDeque as a lifo stack for path exploration.",
        fontsize=11,
    )

    # --- Page 2: ArrayList Deep Dive ---
    p2 = doc.new_page()
    p2.insert_text(
        (40, 50),
        "Chapter 1: ArrayList Deep Dive\n"
        "==============================\n\n"
        "1. What is ArrayList?\n"
        "ArrayList is a resizable, dynamic array implementation of the List interface in Java.\n"
        "Unlike standard arrays with fixed size, an ArrayList automatically grows as elements are added.\n\n"
        "2. Internal Working and Resizing:\n"
        "- Default initial capacity: 10 elements.\n"
        "- Growth factor: When capacity is exceeded, ArrayList creates a new array of size (oldCapacity + (oldCapacity >> 1)),\n"
        "  which is approximately 1.5x the previous capacity, and copies elements using System.arraycopy().\n\n"
        "3. Time Complexity:\n"
        "- Access / Retrieval (get, set): O(1) constant time, because it uses index-based direct memory access.\n"
        "- Insertion at end (amortized): O(1) amortized constant time.\n"
        "- Insertion / Deletion at arbitrary index: O(n) linear time because following elements must be shifted.\n"
        "- Search by value (contains, indexOf): O(n) linear scan.\n\n"
        "4. Key Characteristics:\n"
        "- Maintains insertion order.\n"
        "- Allows null elements and duplicate values.\n"
        "- Not synchronized (use Collections.synchronizedList or CopyOnWriteArrayList for thread safety).\n"
        "- Uses fail-fast iterators that throw ConcurrentModificationException on concurrent structural modification.",
        fontsize=11,
    )

    # --- Page 3: HashMap Deep Dive ---
    p3 = doc.new_page()
    p3.insert_text(
        (40, 50),
        "Chapter 2: HashMap Deep Dive\n"
        "============================\n\n"
        "1. What is HashMap?\n"
        "HashMap is a hash table-based implementation of the Map interface, storing data in key-value pairs (K, V).\n"
        "It provides fast lookups, insertions, and deletions based on the hashCode() and equals() methods of the keys.\n\n"
        "2. Internal Architecture:\n"
        "- Array of Nodes (Buckets): Internally represented as Node<K,V>[] table.\n"
        "- Default Initial Capacity: 16 buckets.\n"
        "- Default Load Factor: 0.75 (threshold = capacity * loadFactor = 12 elements before resizing/rehashing).\n"
        "- Index calculation: index = (n - 1) & hash(key), where n is array length (power of two).\n\n"
        "3. Collision Handling & Treeification (Java 8+):\n"
        "- Chaining: When two keys produce the same bucket index, entries are stored in a linked list.\n"
        "- Treeification Threshold: When a single bucket's linked list length reaches 8 and the table capacity is at least 64,\n"
        "  the bucket is converted from a singly-linked list into a balanced Red-Black Tree (TreeNode).\n"
        "- This improves worst-case lookup from O(n) down to O(log n).\n\n"
        "4. Time Complexity:\n"
        "- put(key, value) and get(key): O(1) average constant time.\n"
        "- Worst-case (severe hash collisions): O(log n) due to Red-Black Tree optimization.\n"
        "- Allows one null key and multiple null values.",
        fontsize=11,
    )

    # --- Page 4: Queue, Deque, Set & Summary Table ---
    p4 = doc.new_page()
    p4.insert_text(
        (40, 50),
        "Chapter 3: Set, Queue, and Time Complexity Summary\n"
        "===================================================\n\n"
        "1. Set Implementations:\n"
        "- HashSet: Backed by an internal HashMap. O(1) operations. No ordering.\n"
        "- LinkedHashSet: Maintains insertion order via a doubly-linked list. O(1) operations.\n"
        "- TreeSet: Backed by a Red-Black Tree (NavigableMap). Maintains sorted natural order. O(log n) operations.\n\n"
        "2. Queue & Deque Implementations:\n"
        "- PriorityQueue: Backed by a min-heap. Provides O(log n) insertion and removal of the highest priority element.\n"
        "- ArrayDeque: Resizable circular array implementation of Deque. O(1) amortized insertion/removal at both ends.\n"
        "  Faster than LinkedList and Stack for BFS traversal and LIFO stack operations.\n\n"
        "3. Quick Time Complexity Reference Table:\n"
        "Data Structure    | Access   | Search   | Insertion | Deletion\n"
        "------------------+----------+----------+-----------+----------\n"
        "ArrayList         | O(1)     | O(n)     | O(1)*     | O(n)\n"
        "LinkedList        | O(n)     | O(n)     | O(1)      | O(1)\n"
        "HashMap           | N/A      | O(1) avg | O(1) avg  | O(1) avg\n"
        "HashSet           | N/A      | O(1) avg | O(1) avg  | O(1) avg\n"
        "TreeSet / TreeMap | N/A      | O(log n) | O(log n)  | O(log n)\n"
        "ArrayDeque        | O(1) ends| O(n)     | O(1) ends | O(1) ends\n"
        "PriorityQueue     | O(1) min | O(n)     | O(log n)  | O(log n)",
        fontsize=11,
    )

    doc.save(str(pdf_path))
    doc.close()
    print(f"[OK] Generated rich handbook: {pdf_path}")

if __name__ == "__main__":
    create_handbook()

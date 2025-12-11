"""
AVL Tree Implementation with ASCII Visualization
"""

class AVLNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    def __init__(self):
        self.root = None
    
    def height(self, node):
        if not node:
            return 0
        return node.height
    
    def balance_factor(self, node):
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)
    
    def update_height(self, node):
        if node:
            node.height = 1 + max(self.height(node.left), self.height(node.right))
    
    # Right rotation
    #       y                x
    #      / \              / \
    #     x   C    -->     A   y
    #    / \                  / \
    #   A   B                B   C
    def rotate_right(self, y):
        x = y.left
        B = x.right
        
        x.right = y
        y.left = B
        
        self.update_height(y)
        self.update_height(x)
        
        return x
    
    # Left rotation
    #     x                  y
    #    / \                / \
    #   A   y      -->     x   C
    #      / \            / \
    #     B   C          A   B
    def rotate_left(self, x):
        y = x.right
        B = y.left
        
        y.left = x
        x.right = B
        
        self.update_height(x)
        self.update_height(y)
        
        return y
    
    def insert(self, key):
        self.root = self._insert(self.root, key)
    
    def _insert(self, node, key):
        # Standard BST insertion
        if not node:
            return AVLNode(key)
        
        if key < node.key:
            node.left = self._insert(node.left, key)
        elif key > node.key:
            node.right = self._insert(node.right, key)
        else:
            return node  # Duplicate keys not allowed
        
        # Update height
        self.update_height(node)
        
        # Get balance factor
        balance = self.balance_factor(node)
        
        # Left Left Case
        if balance > 1 and key < node.left.key:
            print(f"    ⟳ Right rotation at {node.key} (Left-Left case)")
            return self.rotate_right(node)
        
        # Right Right Case
        if balance < -1 and key > node.right.key:
            print(f"    ⟲ Left rotation at {node.key} (Right-Right case)")
            return self.rotate_left(node)
        
        # Left Right Case
        if balance > 1 and key > node.left.key:
            print(f"    ⟲ Left rotation at {node.left.key}, then ⟳ Right rotation at {node.key} (Left-Right case)")
            node.left = self.rotate_left(node.left)
            return self.rotate_right(node)
        
        # Right Left Case
        if balance < -1 and key < node.right.key:
            print(f"    ⟳ Right rotation at {node.right.key}, then ⟲ Left rotation at {node.key} (Right-Left case)")
            node.right = self.rotate_right(node.right)
            return self.rotate_left(node)
        
        return node
    
    def visualize(self):
        """Print ASCII visualization of the tree."""
        if not self.root:
            print("  (empty tree)")
            return
        
        lines, *_ = self._build_tree_string(self.root)
        for line in lines:
            print("  " + line)
    
    def _build_tree_string(self, node):
        """Build ASCII art representation of tree."""
        if not node:
            return [], 0, 0, 0
        
        line = f"({node.key})"
        width = len(line)
        
        # Leaf node
        if not node.left and not node.right:
            return [line], width, 1, width // 2
        
        # Only left child
        if not node.right:
            left_lines, left_width, left_height, left_root = self._build_tree_string(node.left)
            first_line = (left_root + 1) * ' ' + (left_width - left_root - 1) * '_' + line
            second_line = left_root * ' ' + '/' + (left_width - left_root - 1 + width) * ' '
            shifted_lines = [line + (width) * ' ' for line in left_lines]
            return [first_line, second_line] + shifted_lines, left_width + width, left_height + 2, left_width + width // 2
        
        # Only right child
        if not node.left:
            right_lines, right_width, right_height, right_root = self._build_tree_string(node.right)
            first_line = line + right_root * '_' + (right_width - right_root) * ' '
            second_line = (width + right_root) * ' ' + '\\' + (right_width - right_root - 1) * ' '
            shifted_lines = [(width) * ' ' + line for line in right_lines]
            return [first_line, second_line] + shifted_lines, width + right_width, right_height + 2, width // 2
        
        # Both children
        left_lines, left_width, left_height, left_root = self._build_tree_string(node.left)
        right_lines, right_width, right_height, right_root = self._build_tree_string(node.right)
        
        first_line = (left_root + 1) * ' ' + (left_width - left_root - 1) * '_' + line + right_root * '_' + (right_width - right_root) * ' '
        second_line = left_root * ' ' + '/' + (left_width - left_root - 1 + width + right_root) * ' ' + '\\' + (right_width - right_root - 1) * ' '
        
        # Pad shorter side
        if left_height < right_height:
            left_lines += [left_width * ' '] * (right_height - left_height)
        elif right_height < left_height:
            right_lines += [right_width * ' '] * (left_height - right_height)
        
        paired_lines = [a + width * ' ' + b for a, b in zip(left_lines, right_lines)]
        
        return [first_line, second_line] + paired_lines, left_width + width + right_width, max(left_height, right_height) + 2, left_width + width // 2


def main():
    print("=" * 60)
    print("         AVL Tree - Self-Balancing BST Demo")
    print("=" * 60)
    
    tree = AVLTree()
    numbers = [50, 25, 75, 10, 30, 60, 80, 5, 15, 35]
    
    for i, num in enumerate(numbers, 1):
        print(f"\n{'─' * 60}")
        print(f"Step {i}: Insert {num}")
        print('─' * 60)
        tree.insert(num)
        print("\nTree structure:")
        tree.visualize()
    
    print(f"\n{'=' * 60}")
    print("Final AVL Tree Properties:")
    print(f"  Root: {tree.root.key}")
    print(f"  Height: {tree.height(tree.root)}")
    print(f"  Balance factor at root: {tree.balance_factor(tree.root)}")
    print("=" * 60)
    
    # Test with a sequence that causes rotations
    print("\n\n" + "=" * 60)
    print("     AVL Tree - Rotation Demo (Worst-case insertions)")
    print("=" * 60)
    
    tree2 = AVLTree()
    import random
    random.seed(42)  # For reproducibility; remove or change the seed for different results
    ascending = random.sample(range(1001, 10001), 10)
    
    print("\nInserting ascending sequence (would be a linked list in BST):")
    for i, num in enumerate(ascending, 1):
        print(f"\n{'─' * 60}")
        print(f"Step {i}: Insert {num}")
        print('─' * 60)
        tree2.insert(num)
        print("\nTree structure:")
        tree2.visualize()
    
    print(f"\n{'=' * 60}")
    print("Final AVL Tree (balanced despite ascending insertion):")
    print(f"  Root: {tree2.root.key}")
    print(f"  Height: {tree2.height(tree2.root)} (optimal: {int(__import__('math').log2(10)) + 1})")
    print("=" * 60)


if __name__ == "__main__":
    main()


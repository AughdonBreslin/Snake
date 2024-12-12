import numpy as np
import random

from collections import Counter
from math import ceil, log10

class Node():
    def __init__(self, val, pos):
        self.val = val
        self.pos = pos

    def __format__(self, format_spec):
        return f"{self.val:{format_spec}}"

    def __repr__(self):
        return f"{self.val}"
    
    def __hash__(self):
        return hash(self.val)
    
    def __eq__(self, other):
        return self.val == other
    
    def __lt__(self, other):
        return self.val < other.val

    def __iter__(self):
        return iter([self.val])

    
class AdjacencyList():
    def __init__(self):
        self.adj_list = {}

    def __index__(self, node):
        return self.adj_list[node]

    def __repr__(self):
        res = ""
        for node in self.adj_list:
            res += f"{node} {[nod for nod in self.adj_list[node]]}\n"
        return res

    def set_adjs(self, node, adjs):
        self.adj_list[node] = adjs

    def add_adj(self, node, adj):
        if node not in self.adj_list:
            self.adj_list[node] = []
        if adj not in self.adj_list:
            self.adj_list[adj] = []
        self.adj_list[node].append(adj)
        self.adj_list[adj].append(node)

    def remove_adjs(self, node, adj):
        self.adj_list[node].remove(adj)
        self.adj_list[adj].remove(node)
    
    def get(self, node):
        return self.adj_list[node]
    
    def get_random(self, node):
        return random.choice(self.adj_list[node])
    
    def is_adj(self, node, adj):
        return adj in self.adj_list[node] and node in self.adj_list[adj]
    
    def find_ends(self):
        ends = []
        for node in self.adj_list:
            if len(self.adj_list[node]) == 1:
                ends.append(node)
        if not (len(ends) == 0 or len(ends) == 2):
            raise Exception(f"Should be exactly zero or two ends: {ends}")
        return ends
    
    def loops(self, start, branch):
        visited = set(start)
        stack = [branch]
        while stack:
            node = stack.pop()
            visited.add(node)
            adjacencies = self.adj_list[node]
            if len(adjacencies) < 1 or len(adjacencies) > 2:
                raise Exception(f"Unhandled number of adjacencies: {adjacencies}")

            if all(adj in visited for adj in adjacencies):
                if len(adjacencies) == 2:
                    return True
                if len(adjacencies) == 1:
                    return False
            
            if adjacencies[0] not in visited:
                stack.append(adjacencies[0])
            elif adjacencies[1] not in visited:
                stack.append(adjacencies[1])
        
        raise Exception(f"Exhausted search via neither loop nor end?")

class HamiltonianCycle():
    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.transpose = r & 1
        if self.transpose:
            self.r, self.c = self.c, self.r
            self.grid = np.array([[Node(i + j*self.r, (i, j)) for j in range(self.c)] for i in range(self.r)])
        else:
            self.grid = np.array([[Node(i*self.c + j, (i, j)) for j in range(self.c)] for i in range(self.r)])
        self.adj_list = AdjacencyList()

    def _go_left(self, i):
        self.adj_list.set_adjs(self.grid[i+1][self.c-1], [self.grid[i][self.c-1], self.grid[i+1][self.c-2]])
        for j in range(self.c-2, 1, -1):
            self.adj_list.set_adjs(self.grid[i+1][j], [self.grid[i+1][j+1], self.grid[i+1][j-1]])
        
    def _go_right(self, i):
        for j in range(2, self.c-1):
            self.adj_list.set_adjs(self.grid[i][j], [self.grid[i][j-1], self.grid[i][j+1]])
        self.adj_list.set_adjs(self.grid[i][self.c-1], [self.grid[i][self.c-2], self.grid[i+1][self.c-1]])

    def generate_base(self):
        # Top left corner
        self.adj_list.set_adjs(self.grid[0][0], [self.grid[1][0], self.grid[0][1]])
        
        # Start first loop
        if self.c > 2:
            self.adj_list.set_adjs(self.grid[0][1], [self.grid[0][0], self.grid[0][2]])
        self._go_right(0)
        self._go_left(0)

        # Main body
        for i in range(2, self.r-1, 2):
            self.adj_list.set_adjs(self.grid[i-1][1], [self.grid[i-1][2], self.grid[i][1]])
            self.adj_list.set_adjs(self.grid[i][1], [self.grid[i-1][1], self.grid[i][2]])
            self._go_right(i)
            self._go_left(i)
        
        # Bottom left corner
        if self.c > 2:
            self.adj_list.set_adjs(self.grid[self.r-1][1], [self.grid[self.r-1][2], self.grid[self.r-1][0]])
        self.adj_list.set_adjs(self.grid[self.r-1][0], [self.grid[self.r-1][1], self.grid[self.r-2][0]])

        # Come back up
        for i in range(self.r-2, 0, -1):
            self.adj_list.set_adjs(self.grid[i][0], [self.grid[i+1][0], self.grid[i-1][0]])

    def unconnected_adjacencies(self, end):
        i, j = end.pos
        adj = []
        if i > 0 and not self.adj_list.is_adj(self.grid[i][j], self.grid[i-1][j]):
            adj.append(self.grid[i-1][j])
        if i < self.r - 1 and not self.adj_list.is_adj(self.grid[i][j], self.grid[i+1][j]):
            adj.append(self.grid[i+1][j])
        if j > 0 and not self.adj_list.is_adj(self.grid[i][j], self.grid[i][j-1]):
            adj.append(self.grid[i][j-1])
        if j < self.c - 1 and not self.adj_list.is_adj(self.grid[i][j], self.grid[i][j+1]):
            adj.append(self.grid[i][j+1])
        return adj

    # https://arxiv.org/pdf/cond-mat/0508094.pdf
    def backbite(self, ends):
        pos = 0
        choices = [x for x in self.unconnected_adjacencies(ends[pos]) if x not in ends]
        if not choices: # Check the other end
            pos = 1
            choices = [x for x in self.unconnected_adjacencies(ends[pos]) if x not in ends]
            if not choices:
                print(f"Deadlocked :(", end='\r')
                return []
            
        valency3 = random.choice(choices)
        self.adj_list.add_adj(ends[pos], valency3)

        # Find and break the newly created loop
        branches = [x for x in self.adj_list.get(valency3) if x != ends[pos]]
        for branch in branches:
            if self.adj_list.loops(valency3, branch):
                self.adj_list.remove_adjs(valency3, branch)
                return [ends[1-pos], branch]
        
        raise Exception(f"Created new connection but no loop was formed?")

    def attempt_ham_cycle(self):
        # Create a break in base
        ends = self.adj_list.find_ends()
        if not ends:
            ends = [random.choice(list(self.adj_list.adj_list.keys()))]
            ends.append(self.adj_list.get_random(ends[0]))
            self.adj_list.remove_adjs(ends[0], ends[1])
        ends_counter = Counter()
        ends_counter[f'{min(ends[0], ends[1])}, {max(ends[0], ends[1])}'] += 1
        successes = 0
        while ends[0] not in self.unconnected_adjacencies(ends[1]) or successes == 0:
            ends = self.backbite(ends)
            if not ends:
                return None
            ends_counter[f'{min(ends[0], ends[1])}, {max(ends[0], ends[1])}'] += 1
            if len(ends_counter) >= self.r*self.c*self.r*self.c//4: # Empirical limit of all backbite starting positions
                print(ends_counter)
                raise Exception(f"Seems to not be possible.")
            successes += 1
        self.adj_list.add_adj(ends[0], ends[1])
        print(f"\nCycle found in {successes} backbites!")
        return self.grid, self.adj_list

    def randomize_ham_cycle(self):
        fails = 0
        attempt = self.attempt_ham_cycle()
        while not attempt:
            fails += 1
            print(f"Failed attempts: {fails}", end='\r')
            attempt = self.attempt_ham_cycle()
        return attempt

    def get_cycle(self, cycles):
        self.generate_base()
        self.print_path_as_ascii()
        for _ in range(cycles):
            self.grid, self.adj_list = self.randomize_ham_cycle()
            self.print_path_as_ascii()
        if self.transpose:
            self.grid = np.transpose(self.grid)
            self.r, self.c = self.c, self.r
        return self.grid, self.adj_list

    def print_path_as_ascii(self):
        length = ceil(log10(self.r*self.c))
        for i in range(self.r):
            for j in range(self.c):
                print(f"{self.grid[i][j]:0{length}d}", end='')
                if j < self.c - 1:
                    print(' - ' if self.adj_list.is_adj(self.grid[i][j], self.grid[i][j+1]) else '   ', end='')
            print()
            if i < self.r - 1:
                for j in range(self.c):
                    print(f"{' '*(length//2)}|{' '*((length-1)//2)}   " if self.adj_list.is_adj(self.grid[i][j], self.grid[i+1][j]) else f"   {' '*length}", end='')
                print()
    
    def print_cycle_positions(self, start_pos):
        pos = start_pos
        from_pos = start_pos
        print(f"{pos}", end='')
        for _ in range(self.r*self.c-1):
            from_pos = pos
            pos = self.adj_list.get(pos)[0]
            if pos == from_pos:
                pos = self.adj_list.get(pos)[1]
            print(f" -> {pos}", end='')
        print()

if __name__ == "__main__":
    r, c = 3, 20
    cycles = 1

    ham = HamiltonianCycle(r, c)
    grid, adj_list = ham.get_cycle(cycles)
    print("End Result:")
    ham.print_path_as_ascii()
    ham.print_cycle_positions(0)

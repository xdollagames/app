#!/usr/bin/env python3
"""
Seed Finder DISTRIBUTED - Pentru cluster de mașini

Arhitectură:
- Master: distribuie task-uri și colectează rezultate
- Workers: procesează seed ranges și returnează rezultate

Utilizare:
    # Master node
    python3 seed_finder_distributed.py --mode master --workers-file workers.txt --seed-range 0 10000000000
    
    # Worker node
    python3 seed_finder_distributed.py --mode worker --master-ip 192.168.1.100 --master-port 9999

workers.txt format:
    192.168.1.101:64
    192.168.1.102:64
    192.168.1.103:32
    # IP:NUM_CORES
"""

import argparse
import json
import socket
import threading
import time
import pickle
from typing import List, Dict, Tuple
from queue import Queue
import sys

# Import din seed_finder_optimized
from seed_finder_optimized import OptimizedSeedFinder, test_seed_worker, FastLCG, FastXorshift


class DistributedMaster:
    def __init__(self, port: int = 9999):
        self.port = port
        self.workers = []  # [(ip, cores), ...]
        self.task_queue = Queue()
        self.results = []
        self.completed_tasks = 0
        self.total_tasks = 0
        self.lock = threading.Lock()
        self.running = True
    
    def load_workers(self, workers_file: str):
        """Încarcă lista de workers"""
        with open(workers_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ip, cores = line.split(':')
                    self.workers.append((ip, int(cores)))
        
        print(f"✓ Loaded {len(self.workers)} workers")
        total_cores = sum(c for _, c in self.workers)
        print(f"  Total cores: {total_cores}")
    
    def create_tasks(self, seed_start: int, seed_end: int, chunk_size: int = 10000000):
        """Creează task-uri pentru workers"""
        current = seed_start
        task_id = 0
        
        while current < seed_end:
            chunk_end = min(current + chunk_size, seed_end)
            task = {
                'task_id': task_id,
                'seed_start': current,
                'seed_end': chunk_end,
                'timestamp': time.time()
            }
            self.task_queue.put(task)
            current = chunk_end
            task_id += 1
        
        self.total_tasks = task_id
        print(f"✓ Created {self.total_tasks} tasks (chunk size: {chunk_size:,})")
    
    def handle_worker(self, conn, addr):
        """Handle worker connection"""
        print(f"Worker connected: {addr}")
        
        try:
            # Send configuration
            config = pickle.dumps({
                'target_draws': self.target_draws,
                'rng_type': self.rng_type,
                'threshold': self.threshold
            })
            conn.sendall(len(config).to_bytes(8, 'big'))
            conn.sendall(config)
            
            while self.running and not self.task_queue.empty():
                # Get task
                try:
                    task = self.task_queue.get(timeout=1)
                except:
                    continue
                
                # Send task
                task_data = pickle.dumps(task)
                conn.sendall(len(task_data).to_bytes(8, 'big'))
                conn.sendall(task_data)
                
                # Receive result
                result_size = int.from_bytes(conn.recv(8), 'big')
                result_data = b''
                while len(result_data) < result_size:
                    chunk = conn.recv(min(4096, result_size - len(result_data)))
                    if not chunk:
                        break
                    result_data += chunk
                
                result = pickle.loads(result_data)
                
                # Store results
                with self.lock:
                    self.results.extend(result['candidates'])
                    self.completed_tasks += 1
                    
                    # Progress
                    progress = self.completed_tasks / self.total_tasks * 100
                    print(f"\r[{progress:6.2f}%] Tasks: {self.completed_tasks}/{self.total_tasks} | "
                          f"Found: {len(self.results)}", end='', flush=True)
        
        except Exception as e:
            print(f"\nWorker {addr} error: {e}")
        finally:
            conn.close()
    
    def run(self, seed_start: int, seed_end: int, target_draws: List[List[int]],
            rng_type: str = 'lcg', threshold: float = 0.25):
        """Run master server"""
        
        self.target_draws = target_draws
        self.rng_type = rng_type
        self.threshold = threshold
        
        print(f"\n{'='*70}")
        print(f"DISTRIBUTED MASTER")
        print(f"{'='*70}")
        print(f"Listening on port: {self.port}")
        print(f"Seed range: {seed_start:,} → {seed_end:,}")
        
        # Create tasks
        self.create_tasks(seed_start, seed_end)
        
        # Start server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.port))
        server.listen(10)
        
        print(f"\n✓ Master ready, waiting for workers...")
        
        start_time = time.time()
        
        # Accept workers
        threads = []
        try:
            while self.running and self.completed_tasks < self.total_tasks:
                server.settimeout(1.0)
                try:
                    conn, addr = server.accept()
                    t = threading.Thread(target=self.handle_worker, args=(conn, addr))
                    t.daemon = True
                    t.start()
                    threads.append(t)
                except socket.timeout:
                    continue
            
            # Wait for completion
            for t in threads:
                t.join()
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
        finally:
            self.running = False
            server.close()
        
        elapsed = time.time() - start_time
        
        print(f"\n\n{'='*70}")
        print(f"COMPLETE")
        print(f"{'='*70}")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Total results: {len(self.results)}")
        
        # Sort and save
        self.results.sort(key=lambda x: x['avg_score'], reverse=True)
        
        output_file = f"distributed_results_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'seed_range': [seed_start, seed_end],
                'execution_time': elapsed,
                'total_tasks': self.total_tasks,
                'results': self.results[:1000]
            }, f, indent=2)
        
        print(f"💾 Results saved: {output_file}")
        
        return self.results


class DistributedWorker:
    def __init__(self, master_ip: str, master_port: int):
        self.master_ip = master_ip
        self.master_port = master_port
    
    def run(self):
        """Run worker"""
        print(f"\n{'='*70}")
        print(f"DISTRIBUTED WORKER")
        print(f"{'='*70}")
        print(f"Connecting to master: {self.master_ip}:{self.master_port}")
        
        try:
            # Connect to master
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((self.master_ip, self.master_port))
            print("✓ Connected to master")
            
            # Receive configuration
            config_size = int.from_bytes(sock.recv(8), 'big')
            config_data = sock.recv(config_size)
            config = pickle.loads(config_data)
            
            target_draws = config['target_draws']
            rng_type = config['rng_type']
            threshold = config['threshold']
            
            print(f"✓ Configuration received")
            print(f"  Target draws: {len(target_draws)}")
            print(f"  RNG type: {rng_type}")
            print(f"  Threshold: {threshold:.1%}\n")
            
            task_count = 0
            
            # Process tasks
            while True:
                # Receive task
                task_size_bytes = sock.recv(8)
                if not task_size_bytes:
                    break
                
                task_size = int.from_bytes(task_size_bytes, 'big')
                task_data = sock.recv(task_size)
                task = pickle.loads(task_data)
                
                task_count += 1
                print(f"Task {task_count}: Processing seeds {task['seed_start']:,} → {task['seed_end']:,}")
                
                # Process seeds
                results = []
                for seed in range(task['seed_start'], task['seed_end']):
                    result = test_seed_worker(seed, target_draws, rng_type, threshold)
                    if result:
                        results.append(result)
                
                # Send results
                result = {
                    'task_id': task['task_id'],
                    'candidates': results
                }
                result_data = pickle.dumps(result)
                sock.sendall(len(result_data).to_bytes(8, 'big'))
                sock.sendall(result_data)
                
                print(f"  → Found {len(results)} candidates")
        
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            sock.close()
            print("\nDisconnected from master")


def main():
    parser = argparse.ArgumentParser(
        description='Seed Finder DISTRIBUTED - Multi-machine computing'
    )
    parser.add_argument('--mode', type=str, required=True, choices=['master', 'worker'])
    
    # Master arguments
    parser.add_argument('--workers-file', type=str, help='File with worker IPs')
    parser.add_argument('--seed-range', type=int, nargs=2, metavar=('START', 'END'))
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--input', type=str, default='/app/backend/loto_data.json')
    parser.add_argument('--draws', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--rng', type=str, default='lcg', choices=['lcg', 'xorshift'])
    
    # Worker arguments
    parser.add_argument('--master-ip', type=str, help='Master IP address')
    parser.add_argument('--master-port', type=int, default=9999)
    
    args = parser.parse_args()
    
    if args.mode == 'master':
        if not args.workers_file or not args.seed_range:
            print("Error: Master needs --workers-file and --seed-range")
            return
        
        finder = OptimizedSeedFinder(args.input)
        target_draws = [finder.draws[i]['numbers_sorted']
                       for i in range(min(args.draws, len(finder.draws)))]
        
        master = DistributedMaster(args.port)
        master.load_workers(args.workers_file)
        results = master.run(
            seed_start=args.seed_range[0],
            seed_end=args.seed_range[1],
            target_draws=target_draws,
            rng_type=args.rng,
            threshold=args.threshold
        )
        
        if results:
            print(f"\nTOP 10:")
            for i, r in enumerate(results[:10], 1):
                print(f"{i:2}. Seed {r['seed']:<12,}: {r['avg_score']:.2%}")
    
    elif args.mode == 'worker':
        if not args.master_ip:
            print("Error: Worker needs --master-ip")
            return
        
        worker = DistributedWorker(args.master_ip, args.master_port)
        worker.run()


if __name__ == '__main__':
    main()

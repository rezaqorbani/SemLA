import argparse
import json
import os
import yaml
import scipy
import scipy.spatial
from typing import Dict, List, Callable, Any, Optional, Tuple

from domain_orchestrator.domain_orchestrator import DomainOrchestrator

# Define distance measure mappings
NAME_MEASURE_MAPPING = {
    "euclidean": lambda u, v: 1. / scipy.spatial.distance.euclidean(u.squeeze(), v.squeeze()),
    "cosine": lambda u, v: scipy.spatial.distance.cosine(u.squeeze(), v.squeeze()),
}

def load_domains_from_yaml(file_path: str) -> List[str]:
    """Load domains from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, weights: Optional[Dict] = None, output_dir: str = "./results") -> None:
    """Save results and weights to JSON files."""
    
    # Change the current working directory to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(f"Changing current working directory to {root_dir}")
    os.chdir(root_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    if weights is not None:
        with open(os.path.join(output_dir, "weights.json"), 'w') as f:
            json.dump(weights, f, indent=4)
    
    print(f"Results saved to {output_dir}")

def benchmark_zeroshot(source_domains: List[str], target_domains: List[str], 
                      output_dir: str) -> None:
    """Run zero-shot benchmark experiment."""
    orchestrator = DomainOrchestrator(source_domains)
    results = orchestrator.benchmark_zeroshot(target_domains)
    save_results(results, output_dir=output_dir)

def benchmark_oracle(source_domains: List[str], target_domains: List[str], 
                    output_dir: str) -> None:
    """Run oracle benchmark experiment."""
    orchestrator = DomainOrchestrator(domains=source_domains)
    results = orchestrator.benchmark_oracle(target_domains=target_domains)
    save_results(results, output_dir=output_dir)

def uniform_merge(source_domains: List[str], target_domains: List[str], 
                 remove_target_adapter: bool, output_dir: str) -> None:
    """Run uniform merge experiment."""
    orchestrator = DomainOrchestrator(domains=source_domains)
    results, weights = orchestrator.benchmark_uniform(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
    )
    save_results(results, weights, output_dir=output_dir)

def semla_merge(source_domains: List[str], target_domains: List[str], 
                config: Dict[str, Any], remove_target_adapter: bool, 
                output_dir: str) -> None:
    """Run online merge experiment."""
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    temperature = config.get("temperature", 0.05)
    top_k = config.get("top_k", 5)
    combination_type = config.get("combination_type", "cat")
    
    orchestrator = DomainOrchestrator(source_domains)
    results, weights = orchestrator.benchmark_semla(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
        similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
        softmax_temperature=temperature,
        top_k=top_k,
        combination_type=combination_type
    )
    save_results(results, weights, output_dir=output_dir)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Domain adaptation experiments")
    
    # Required arguments
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["zeroshot", "oracle", "uniform", "semla"],
                        help="Type of experiment to run")
    
    # Optional arguments with defaults
    parser.add_argument("--source_domains", type=str, 
                        help="Path to YAML file containing source domains")
    parser.add_argument("--target_domains", type=str, 
                        help="Path to YAML file containing target domains")
    parser.add_argument("--semla_config", type=str, 
                        help="Path to YAML file containing configuration parameters")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--remove_target_adapter", action="store_true", 
                        help="Whether to remove target adapter")
    
    return parser.parse_args()

def main():
    """Main function to run experiments based on command line arguments."""

    args = parse_args()
    
    # Load source domains
    source_domains = load_domains_from_yaml(args.source_domains) if args.source_domains else []
    
    # Load target domains
    target_domains = load_domains_from_yaml(args.target_domains) if args.target_domains else []
    
    # Load config if provided
    semla_config = load_config_from_yaml(args.semla_config) if args.semla_config else {}
    
    # Run the specified experiment
    if args.experiment == "zeroshot":
        benchmark_zeroshot(source_domains, target_domains, args.output_dir)
    elif args.experiment == "oracle":
        benchmark_oracle(source_domains, target_domains, args.output_dir)
    elif args.experiment == "uniform":
        uniform_merge(source_domains, target_domains, args.remove_target_adapter, args.output_dir)
    elif args.experiment == "semla":
        semla_merge(source_domains, target_domains, semla_config, args.remove_target_adapter, args.output_dir)

if __name__ == "__main__":
    main()
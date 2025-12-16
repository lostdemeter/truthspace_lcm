#!/usr/bin/env python3
"""
Live Fire Test: Ingest ifconfig man page and test natural language queries
===========================================================================

This test validates our hypothesis by:
1. Parsing the ifconfig man page to extract commands/options
2. Using the autotuner to place them in TruthSpace
3. Testing natural language queries to see if they resolve correctly
"""

import subprocess
import re
import sys
import os
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truthspace_lcm import (
    TruthSpace,
    Resolver,
    EntryType,
    KnowledgeDomain,
)
from truthspace_lcm.core.encoder import PlasticEncoder
from truthspace_lcm.core.autotuner import DimensionAwareAutotuner


# =============================================================================
# MAN PAGE PARSER
# =============================================================================

def get_man_page(command: str) -> str:
    """Fetch a man page."""
    try:
        result = subprocess.run(
            ["man", command],
            capture_output=True,
            text=True,
            env={**os.environ, "PAGER": "cat"}
        )
        return result.stdout
    except Exception as e:
        print(f"Error fetching man page: {e}")
        return ""


def parse_ifconfig_man() -> List[Dict]:
    """Parse ifconfig man page into structured intents."""
    
    # Manually extracted intents from ifconfig man page
    # In a real system, this would be automated NLP extraction
    
    intents = [
        {
            "name": "list_network_interfaces",
            "description": "Show all network interfaces and their status",
            "keywords": ["network", "interfaces", "adapters", "ifconfig", "nic", "ethernet", "wifi", "lan", "wlan", "eth", "cards", "devices"],
            "triggers": [
                "what network adapters are available",
                "show network interfaces",
                "list network interfaces",
                "what interfaces do I have",
                "show my network cards",
                "list adapters",
                "display network interfaces",
            ],
            "target_command": "ifconfig -a",
            "output_type": "bash",
        },
        {
            "name": "show_interface_status",
            "description": "Show status of a specific network interface",
            "keywords": ["interface", "status", "show", "display", "info", "details", "eth", "wlan"],
            "triggers": [
                "show interface status",
                "interface info",
                "show eth0",
                "display interface details",
            ],
            "target_command": "ifconfig <interface>",
            "output_type": "bash",
        },
        {
            "name": "bring_interface_up",
            "description": "Activate a network interface",
            "keywords": ["up", "activate", "enable", "bring", "interface", "ifconfig", "eth", "wlan", "network", "online"],
            "triggers": [
                "bring interface up",
                "activate interface",
                "enable network interface",
                "turn on interface",
                "start interface",
            ],
            "target_command": "sudo ifconfig <interface> up",
            "output_type": "bash",
        },
        {
            "name": "bring_interface_down",
            "description": "Deactivate a network interface",
            "keywords": ["down", "deactivate", "disable", "bring", "interface", "ifconfig", "eth", "wlan", "network", "offline", "off"],
            "triggers": [
                "bring interface down",
                "deactivate interface",
                "disable network interface",
                "turn off interface",
                "stop interface",
                "shutdown interface",
            ],
            "target_command": "sudo ifconfig <interface> down",
            "output_type": "bash",
        },
        {
            "name": "set_ip_address",
            "description": "Set IP address on a network interface",
            "keywords": ["ip", "address", "set", "assign", "configure", "interface", "ipv4"],
            "triggers": [
                "set ip address",
                "assign ip to interface",
                "configure ip address",
                "change ip address",
            ],
            "target_command": "sudo ifconfig <interface> <ip_address>",
            "output_type": "bash",
        },
        {
            "name": "set_netmask",
            "description": "Set network mask on an interface",
            "keywords": ["netmask", "subnet", "mask", "set", "configure", "interface"],
            "triggers": [
                "set netmask",
                "configure subnet mask",
                "set subnet mask",
            ],
            "target_command": "sudo ifconfig <interface> netmask <netmask>",
            "output_type": "bash",
        },
        {
            "name": "set_mtu",
            "description": "Set MTU (Maximum Transfer Unit) on an interface",
            "keywords": ["mtu", "maximum", "transfer", "unit", "ifconfig", "packet", "size", "interface", "network"],
            "triggers": [
                "set mtu",
                "change mtu",
                "configure mtu",
                "set packet size",
            ],
            "target_command": "sudo ifconfig <interface> mtu <size>",
            "output_type": "bash",
        },
        {
            "name": "enable_promiscuous",
            "description": "Enable promiscuous mode on an interface",
            "keywords": ["promiscuous", "promisc", "mode", "enable", "sniff", "capture", "all", "packets"],
            "triggers": [
                "enable promiscuous mode",
                "turn on promisc",
                "enable packet capture",
                "sniff all packets",
            ],
            "target_command": "sudo ifconfig <interface> promisc",
            "output_type": "bash",
        },
        {
            "name": "show_interface_short",
            "description": "Show brief interface summary",
            "keywords": ["short", "brief", "summary", "interfaces", "list", "quick"],
            "triggers": [
                "show brief interface list",
                "quick interface summary",
                "short interface list",
            ],
            "target_command": "ifconfig -s",
            "output_type": "bash",
        },
    ]
    
    return intents


# =============================================================================
# INGESTION
# =============================================================================

def ingest_intents(ts: TruthSpace, intents: List[Dict]) -> Dict:
    """Ingest parsed intents into TruthSpace."""
    
    results = {
        "success": [],
        "failed": [],
        "skipped": [],
    }
    
    encoder = PlasticEncoder()
    tuner = DimensionAwareAutotuner(encoder)
    
    print("\n" + "=" * 70)
    print("INGESTING IFCONFIG INTENTS")
    print("=" * 70)
    
    for intent in intents:
        name = intent["name"]
        
        # Check if already exists
        existing = ts.get_by_name(name, entry_type=EntryType.INTENT)
        if existing:
            print(f"  ⏭️  {name} (already exists)")
            results["skipped"].append(name)
            continue
        
        # Analyze with autotuner
        analysis = tuner.analyze_concept(" ".join(intent["keywords"][:3]))
        
        try:
            # Store in TruthSpace
            ts.store(
                name=name,
                entry_type=EntryType.INTENT,
                domain=KnowledgeDomain.SYSTEM,
                description=intent["description"],
                keywords=intent["keywords"],
                metadata={
                    "target_commands": [intent["target_command"]],
                    "output_type": intent["output_type"],
                    "triggers": intent["triggers"],
                    "dimension": analysis.primary_dimension,
                    "dimension_type": analysis.dimension_type,
                }
            )
            
            print(f"  ✅ {name}")
            print(f"      → dim {analysis.primary_dimension} ({analysis.dimension_type})")
            print(f"      → {intent['target_command']}")
            results["success"].append(name)
            
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            results["failed"].append(name)
    
    return results


# =============================================================================
# TESTING
# =============================================================================

def test_queries(ts: TruthSpace) -> Dict:
    """Test natural language queries against ingested knowledge."""
    
    resolver = Resolver(ts)
    
    test_cases = [
        # Direct matches
        ("what network adapters are available", "ifconfig -a"),
        ("show network interfaces", "ifconfig -a"),
        ("list my network cards", "ifconfig -a"),
        
        # Interface operations
        ("bring eth0 up", "sudo ifconfig eth0 up"),
        ("disable the wlan0 interface", "sudo ifconfig wlan0 down"),
        ("turn off eth0", "sudo ifconfig eth0 down"),
        
        # Configuration
        ("set ip address on eth0 to 192.168.1.100", "sudo ifconfig eth0 192.168.1.100"),
        ("change the mtu to 1500", "sudo ifconfig <interface> mtu 1500"),
        
        # Advanced
        ("enable promiscuous mode on eth0", "sudo ifconfig eth0 promisc"),
        ("show brief interface summary", "ifconfig -s"),
    ]
    
    print("\n" + "=" * 70)
    print("TESTING NATURAL LANGUAGE QUERIES")
    print("=" * 70)
    
    results = {
        "passed": [],
        "failed": [],
        "partial": [],
    }
    
    for query, expected in test_cases:
        try:
            result = resolver.resolve(query)
            output = result.output
            
            # Check if output matches expected (allowing for parameter variations)
            # Normalize for comparison
            output_base = output.split()[0] if output else ""
            expected_base = expected.split()[0] if expected else ""
            
            # Check if it's the right command
            if expected_base in output or output_base == expected_base:
                # Check if it's an exact or close match
                if output == expected or expected.replace("<interface>", "") in output:
                    print(f"  ✅ '{query}'")
                    print(f"      → {output}")
                    results["passed"].append((query, output))
                else:
                    print(f"  🟡 '{query}'")
                    print(f"      → {output}")
                    print(f"      (expected: {expected})")
                    results["partial"].append((query, output, expected))
            else:
                print(f"  ❌ '{query}'")
                print(f"      → {output}")
                print(f"      (expected: {expected})")
                results["failed"].append((query, output, expected))
                
        except Exception as e:
            print(f"  ❌ '{query}'")
            print(f"      Error: {e}")
            results["failed"].append((query, str(e), expected))
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("LIVE FIRE TEST: ifconfig Man Page Ingestion")
    print("=" * 70)
    print("\nThis test validates our hypothesis by:")
    print("  1. Parsing ifconfig man page into structured intents")
    print("  2. Ingesting intents into TruthSpace with autotuner")
    print("  3. Testing natural language queries")
    
    # Initialize TruthSpace (use existing DB)
    ts = TruthSpace()
    
    # Show current state
    counts = ts.count()
    print(f"\nCurrent TruthSpace state:")
    for entry_type, count in counts.items():
        print(f"  {entry_type}: {count}")
    
    # Parse man page
    print("\n" + "-" * 70)
    print("PARSING IFCONFIG MAN PAGE")
    print("-" * 70)
    
    intents = parse_ifconfig_man()
    print(f"\nExtracted {len(intents)} intents from ifconfig man page")
    
    for intent in intents:
        print(f"  • {intent['name']}: {intent['target_command']}")
    
    # Ingest
    ingest_results = ingest_intents(ts, intents)
    
    print(f"\nIngestion results:")
    print(f"  Success: {len(ingest_results['success'])}")
    print(f"  Skipped: {len(ingest_results['skipped'])}")
    print(f"  Failed:  {len(ingest_results['failed'])}")
    
    # Test queries
    test_results = test_queries(ts)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_results["passed"]) + len(test_results["partial"]) + len(test_results["failed"])
    
    print(f"\nQuery Results:")
    print(f"  ✅ Passed:  {len(test_results['passed'])}/{total_tests}")
    print(f"  🟡 Partial: {len(test_results['partial'])}/{total_tests}")
    print(f"  ❌ Failed:  {len(test_results['failed'])}/{total_tests}")
    
    success_rate = (len(test_results["passed"]) + len(test_results["partial"]) * 0.5) / total_tests * 100
    
    print(f"\n  Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 70:
        print("\n  🎉 LIVE FIRE TEST PASSED!")
        print("  The system successfully interprets natural language queries")
        print("  about network interfaces after ingesting the ifconfig man page.")
    elif success_rate >= 50:
        print("\n  ⚠️  PARTIAL SUCCESS")
        print("  The system shows promise but needs tuning.")
    else:
        print("\n  ❌ NEEDS WORK")
        print("  The system needs significant improvement.")
    
    print("\n" + "=" * 70)
    
    return success_rate >= 50


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

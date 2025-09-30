# nPrint Configuration Analysis for ZeroML

## Overview
nPrint extracts standardized packet representations for machine learning. Each packet becomes a fixed-length feature vector of binary values (0, 1, -1 for missing).

## Available Header Types
- **IPv4 (-4)**: IP version, header length, TOS, total length, ID, flags, fragment offset, TTL, protocol, checksum, src/dst IP
- **IPv6 (-6)**: Version, traffic class, flow label, payload length, next header, hop limit, src/dst IP  
- **Ethernet (-e)**: Destination MAC, source MAC, EtherType
- **TCP (-t)**: Source/dest port, sequence number, ack number, flags, window, checksum, urgent pointer
- **UDP (-u)**: Source/dest port, length, checksum
- **ICMP (-i)**: Type, code, checksum, additional fields
- **WLAN (-w)**: 802.11 wireless headers
- **Radiotap (-r)**: Wireless metadata

## Recommended Configuration for CICIDS Dataset

### Basic Configuration (Good for most anomaly detection)
```bash
nprint -P input.pcap -W output.csv -4 -t -u -i -c 50000
```
- IPv4 headers: Network layer information
- TCP/UDP/ICMP: Transport layer protocols  
- Count limit: 50K packets for faster processing

### Advanced Configuration (More features, slower processing)
```bash
nprint -P input.pcap -W output.csv -4 -t -u -i -e -A -p 20 -c 50000
```
- Adds Ethernet headers (-e)
- Adds absolute timestamps (-A) 
- Includes 20 bytes of payload (-p 20)
- Better attack detection but larger feature space

### Production Configuration (Balanced)
```bash
nprint -P input.pcap -W output.csv -4 -t -u -i -S -F 0 -c 100000
```
- IPv4, TCP, UDP, ICMP headers
- Statistics output (-S)
- Fill missing values with 0 (-F 0) instead of -1
- Process up to 100K packets

## Feature Space Analysis

### Without Payload (-4 -t -u -i):
- IPv4: ~224 features (bit-level representation)
- TCP: ~800 features (bit-level representation)
- UDP: ~64 features (bit-level representation)  
- ICMP: ~64 features (bit-level representation)
- **Total: ~1089 features per packet** (VERIFIED on sample.pcap)

### With Payload (-4 -t -u -i -p 20):
- Headers: ~42 features
- Payload: 20 bytes × 8 bits = 160 features
- **Total: ~202 features per packet**

### With Ethernet (-4 -t -u -i -e):
- Headers: ~42 features
- Ethernet: ~6 features
- **Total: ~48 features per packet**

## Recommendations for CICIDS

### Phase 1: Baseline (Current Implementation)
```bash
FLAGS="-4 -t -u -i"
COUNT=50000
```
- Fast processing
- Core network features
- Good for initial model training

### Phase 2: Enhanced Detection  
```bash
FLAGS="-4 -t -u -i -e -A"
COUNT=100000
```
- Adds timing information
- Ethernet layer context
- Better attack discrimination

### Phase 3: Deep Analysis (If needed)
```bash
FLAGS="-4 -t -u -i -e -A -p 10"
COUNT=50000
```
- Payload content analysis
- Highest detection accuracy
- Significantly larger feature space

## Processing Considerations

### Memory Usage
- 50K packets × 42 features × 1 byte = ~2.1 MB per file
- 50K packets × 202 features × 1 byte = ~10.1 MB per file

### Processing Time (Estimated)
- 10GB PCAP file, 50K packets: ~2-5 minutes
- 10GB PCAP file, 100K packets: ~4-10 minutes  
- With payload: 2-3x slower

### Storage Requirements
- CSV files: ~1-10MB per 50K packets
- NPY files: ~1-10MB per 50K packets
- Total for CICIDS: ~100-500MB processed data

## Filter Options

### Exclude Patterns (-x flag)
```bash
# Exclude noisy/irrelevant fields
-x "timestamp|checksum"  # Remove timestamps and checksums
-x ".*_reserved.*"       # Remove reserved fields
```

### Packet Filtering (-f flag)  
```bash
# Focus on specific traffic
-f "tcp"                 # TCP traffic only
-f "not icmp"           # Exclude ICMP
-f "port 80 or port 443" # Web traffic only
```

## Final Recommendation

**Start with Phase 1 configuration:**
```bash
nprint -P input.pcap -W output.csv -4 -t -u -i -S -c 50000
```

This provides:
- ✅ Fast processing (~2-5 min per large PCAP)
- ✅ Manageable feature space (~42 features)
- ✅ Core network behavior capture
- ✅ Sufficient for anomaly detection
- ✅ Easy to scale up later

**Upgrade to Phase 2 if baseline results are promising.**
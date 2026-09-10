IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any draw.io-Network-validation tasks.

## [Project Context]
domain:network-topology-diagrams|env:draw.io-plugin|role:network-infrastructure-architect|task:validate-network-topologies+enforce-segmentation-and-redundancy-rules|loop:detect-errors→output-corrections→trigger-redraw

## [Docs Index]
network-reference:{docs/network-topology-expert.md}
plugin-src:{src/graph-parser.ts,src/validation-engine.ts,src/auto-layout.ts}
*always-read-network-specs-before-validating-graph*

## [Domain Rules + Patterns]
three-tier-model:Core(high-speed-backbone)>Distribution(policy+ACLs+inter-VLAN)>Access(end-devices)
segmentation:DMZ=public-facing-services-isolated-by-firewall|VLANs=distinct-containers/swimlanes|inter-VLAN→requires-L3-switch/router/firewall
wan-boundary:external-traffic-MUST-terminate-at-Firewall-or-Edge-Router|prevent:direct-WAN-to-LAN
dmz-isolation:DMZ-subnet-separated-from-internal-LAN-by-firewall|public-services(web-servers,DNS)→placed-in-DMZ
redundancy:critical-trunk-links-require-dual-homed-connections(LAG/port-channel)|core-switches-should-be-HA-pairs
firewall-policy:all-inbound-traffic-filtered|prevent:bypass-of-firewall-inspection|ACLs-enforced-at-distribution-layer

## [Project Conventions]
topology:WAN-at-top→DMZ→Core→Distribution→Access(top-to-bottom-vertical-bands)|traffic-flows-top→down-through-tiers
routing:lines-terminate-at-device-interface-boundary|orthogonal-routing-for-physical-links|prevent:overlapping-trunk-lines
arrows:solid=physical-link(copper/fiber)|dashed=logical-link/tunnel(VPN,VXLAN,GRE)|prevent:floating-edges
labels:device-hostname+IP-range-on-device-node|interface-names(e.g.,-Gi0/1)-on-edge-endpoints|VLAN-ID-on-container-headers

## [Anti-Patterns]
direct-wan-to-lan|correction:WAN-device-directly-to-workstation→insert-firewall-between-WAN-and-LAN
orphan-device|correction:network-device-with-no-connections→connect-to-appropriate-switch-port-or-remove
vlan-leak|correction:cross-VLAN-connection-without-L3-device→add-L3-switch/router-for-inter-VLAN-routing
single-trunk|correction:single-uplink-between-tiers→add-redundant-link(LAG/port-channel)
missing-dmz|correction:public-services-in-LAN-segment→move-to-isolated-DMZ-subnet-behind-firewall
flat-network|correction:no-segmentation-detected→implement-VLANs-for-broadcast-domain-isolation
missing-redundancy|correction:single-core-switch-without-failover→add-HA-pair-with-HSRP/VRRP
firewall-bypass|correction:traffic-bypassing-firewall-inspection→enforce-ACLs-and-route-through-firewall
missing-wan-firewall|correction:WAN-edge-without-firewall→insert-firewall-at-WAN-perimeter
daisy-chained-access-switches|correction:access-switches-cascaded-serially→star-topology-from-distribution

## [Visual Styling]
icon-style:enforce-Cisco-standard-device-icons|shape=mxgraph.cisco-for-devices|80×60-icon-size
edge-style:enforce-orthogonal|edgeStyle=orthogonalEdgeStyle|solid-for-physical+dashed-for-logical/tunnel
color-palette:WAN-container=#FFE0B2(orange)|DMZ-container=#FFCDD2(red-tint)|LAN-container=#C8E6C9(green-tint)|Core=#E3F2FD(blue-tint)|Distribution=#F3E5F5(purple-tint)|Access=#FFF9C4(yellow-tint)
spacing:tiers-1200px-wide|300px-tall|stacked-vertically|devices-spaced-160px-horizontally|40px-padding-inside-containers

## [Standard Device Icons & Conventions]
firewall:Cisco-firewall-style|placed-at-WAN-perimeter-and-DMZ-boundary
switch:Cisco-workgroup(L2)-or-L3-switch-style|Core=L3|Distribution=L3|Access=L2
router:Cisco-router-style|placed-at-WAN-edge-or-inter-VLAN-gateway
server:standard-rack/tower-server|placed-in-DMZ-or-Access-layer
workstation:PC/laptop-style|placed-in-Access-layer
wireless-ap:AP-style|connected-to-Access-layer-switch

## [Validator Rules Reference & Troubleshooting]

### 1. `DIRECT_WAN_TO_LAN`
- **Trigger**: An edge connects an external WAN/Internet node directly to a private workstation or LAN server, bypassing firewall inspection.
- **Troubleshooting**: Reroute the WAN connection to the Firewall external interface, and route the internal LAN segment from the Firewall internal interface.

### 2. `ORPHAN_DEVICE`
- **Trigger**: A network device (switch, router, firewall, or host) has no connections (0 connected edges).
- **Troubleshooting**: Connect the device to the appropriate switch port or upstream gateway, or remove the device from the diagram if obsolete.

### 3. `VLAN_LEAK`
- **Trigger**: A direct edge connects two nodes inside different VLAN containers (e.g., `VLAN 10` to `VLAN 20`) without routing through a Layer 3 router or firewall node.
- **Troubleshooting**: Reroute the connection through a gateway switch or router interface that handles inter-VLAN routing.

### 4. `REDUNDANCY_WARNING`
- **Trigger**: Critical trunk links between Core Switches/Routers and Distribution Switches have only a single link (missing high-availability dual-homed connections). (Generates a warning).
- **Troubleshooting**: Add a second redundant physical/logical connection between the two devices to establish a port channel/LAG.

### 5. `MISSING_DMZ`
- **Trigger**: Public-facing services (web servers, DNS servers) are placed directly in the LAN segment without a DMZ boundary.
- **Troubleshooting**: Create a DMZ container/subnet and move public-facing services into it. Ensure the DMZ is separated from the LAN by a firewall.

### 6. `FLAT_NETWORK`
- **Trigger**: All devices reside in a single broadcast domain with no VLAN segmentation detected.
- **Troubleshooting**: Implement VLAN segmentation to isolate broadcast domains. Group devices by function (servers, workstations, management) into separate VLANs.

### 7. `FIREWALL_BYPASS`
- **Trigger**: Traffic from an external source reaches internal resources without passing through a firewall node in the path.
- **Troubleshooting**: Ensure all external-to-internal paths route through a firewall. Add ACL enforcement at the distribution layer as defense-in-depth.

### 8. `SINGLE_TRUNK`
- **Trigger**: A single uplink connects two tier devices (e.g., Core to Distribution) without a redundant backup link.
- **Troubleshooting**: Add a second physical or logical link between the devices. Configure as a LAG/port-channel for failover and load balancing.

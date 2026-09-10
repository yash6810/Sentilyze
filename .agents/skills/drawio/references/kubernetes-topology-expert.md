IMPORTANT: Prefer retrieval-led reasoning over pre-training-led reasoning for any draw.io-K8s-validation tasks.

## [Project Context]
domain:kubernetes-topology-diagrams|env:draw.io-plugin|role:kubernetes-expert-reviewer-agent|task:validate-k8s-topology+enforce-resource-hierarchy-and-networking-rules|loop:detect-errors→output-corrections→trigger-redraw

## [Docs Index]
k8s-reference:{docs/kubernetes-topology-expert.md}
plugin-src:{src/graph-parser.ts,src/validation-engine.ts,src/auto-layout.ts}
*always-read-k8s-specs-before-validating-graph*

## [Domain Rules + Patterns]
resource-hierarchy:Cluster>Namespace>Deployment>Pod|enforce:strict-nesting-per-K8s-resource-model
namespace-scoped:Deployment|Pod|Service|ConfigMap|Secret|PVC→must-reside-inside-Namespace-container
cluster-scoped:PersistentVolume|ClusterRole|Namespace→reside-at-Cluster-level-or-outside-Namespaces
service-discovery:Ingress→routes-to-Service→routes-to-Pod/Deployment|prevent:Ingress-direct-to-Pod
networking:pod-to-pod-allowed-within-namespace|cross-namespace-requires-Service-DNS-or-NetworkPolicy
config-binding:ConfigMap+Secret→must-reside-in-same-Namespace-as-referencing-Pods
storage-binding:PVC(namespace-scoped)→binds-to-PV(cluster-scoped)|or-uses-StorageClass-for-dynamic-provisioning
sidecar-pattern:init-containers+sidecars→modeled-as-distinct-nodes-inside-Pod-container
external-services:databases+third-party-APIs→placed-outside-Cluster-boundary

## [Project Conventions]
topology:Cluster-at-top→Namespaces-as-horizontal-lanes→Deployments-stacked-within→Pods-as-horizontal-groups
routing:Ingress→Service→Pod-flow-is-top-to-bottom|lines-terminate-at-resource-boundary|orthogonal-routing
arrows:solid=network-traffic(HTTP/gRPC/TCP)|dashed=volume-mounts/config-refs/storage-bindings|prevent:floating-edges
labels:resource-type-prefix(e.g.,"Deployment:-api-server","Service:-frontend")|annotation-labels-below-icon

## [Anti-Patterns]
orphan-pod|correction:Pod-not-managed-by-Deployment/ReplicaSet/StatefulSet→wrap-in-Deployment
service-without-target|correction:Service-with-no-selector-match→verify-pod-labels-match-Service-selector
ingress-bypass|correction:Ingress-directly-to-Pod→route-through-Service-layer
pvc-without-pv|correction:PVC-not-bound-to-PV→create-PV-or-configure-StorageClass-for-dynamic-provisioning
namespace-leak|correction:cross-namespace-direct-connection→use-Service-DNS({svc}.{ns}.svc.cluster.local)
missing-hpa|correction:Deployment-without-autoscaling-for-variable-load→consider-adding-HorizontalPodAutoscaler
missing-resource-limits|correction:Pod-without-resource-requests/limits→add-ResourceQuota-or-LimitRange
privileged-container|correction:container-running-as-root-without-justification→add-SecurityContext(runAsNonRoot:true)
configmap-cross-namespace|correction:Pod-referencing-ConfigMap-in-different-namespace→move-ConfigMap-to-Pod's-namespace

## [Visual Styling]
icon-style:enforce-K8s-standard-icons|use-shape=mxgraph.kubernetes-for-resources|60×60-icon-size
edge-style:enforce-orthogonal|edgeStyle=orthogonalEdgeStyle|solid-for-traffic+dashed-for-config-refs
color-palette:Cluster=#326CE5(K8s-blue)|Namespace=#E8F0FE(light-blue)|Deployment=#C8E6C9(green-tint)|Service=#BBDEFB(blue-tint)|Pod=#FFF9C4(yellow-tint)|Ingress=#F8BBD0(pink-tint)
spacing:Namespaces-540px-wide|Deployments-stacked-at-parentWidth-40|Pods-spaced-120px-horizontally|60px-padding-inside-containers

## [Validator Rules Reference & Troubleshooting]

### 1. `ORPHAN_POD`
- **Trigger**: A pod node (`type === 'pod'`) is placed outside a Deployment or Namespace container (e.g., directly in the root '1').
- **Troubleshooting**: Nest the pod inside a valid Deployment container, or at minimum inside a Namespace. Bare pods should be wrapped in a Deployment for production diagrams.

### 2. `SERVICE_WITHOUT_TARGET`
- **Trigger**: A Service node (`type === 'service'`) has no outgoing connections to a Pod or Deployment.
- **Troubleshooting**: Connect the Service to the target Pod/Deployment with a solid connector line. Verify that the label selector matches.

### 3. `INGRESS_BYPASS`
- **Trigger**: An Ingress node (`type === 'ingress'`) connects directly to a Pod node, bypassing the Service layer.
- **Troubleshooting**: Reroute the edge from the Ingress to the Service, and then from the Service to the Pod.

### 4. `PVC_WITHOUT_PV`
- **Trigger**: A PersistentVolumeClaim (`type === 'pvc'`) exists but is not connected to a PersistentVolume (`type === 'pv'`) or does not reference a StorageClass.
- **Troubleshooting**: Connect the PVC to a PV using a solid connection edge, or annotate with the StorageClass name for dynamic provisioning.

### 5. `NAMESPACE_LEAK`
- **Trigger**: Direct cross-talk or connections between namespace-scoped resources (like pods or configmaps) in different namespaces without routing through a Service or Ingress.
- **Troubleshooting**: Use Services for cross-namespace communication (`{svc}.{ns}.svc.cluster.local`) or restrict connections using NetworkPolicies.

### 6. `MISSING_RESOURCE_LIMITS`
- **Trigger**: A Pod or container node has no `resources.requests` or `resources.limits` annotations visible in its label.
- **Troubleshooting**: Add resource requests and limits to the Pod spec, or apply a LimitRange to the Namespace.

### 7. `PRIVILEGED_CONTAINER`
- **Trigger**: A container node is labeled as running with `privileged: true` or `runAsUser: 0` without explicit justification.
- **Troubleshooting**: Add a SecurityContext with `runAsNonRoot: true` and `readOnlyRootFilesystem: true` unless the workload requires elevated privileges.

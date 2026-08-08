import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
  type Simulation,
  type SimulationLinkDatum,
  type SimulationNodeDatum,
} from "d3-force-3d";

export type NodeId = string | number;

export interface GraphNode<Id extends NodeId = NodeId> extends SimulationNodeDatum {
  id: Id;
  [key: string]: unknown;
}

export interface GraphLink<Node extends GraphNode = GraphNode> extends SimulationLinkDatum<Node> {
  source: Node["id"] | Node;
  target: Node["id"] | Node;
  [key: string]: unknown;
}

export interface GraphData<Node extends GraphNode, Link extends GraphLink<Node>> {
  nodes: readonly Node[];
  links: readonly Link[];
}

export interface MutableGraph<Node extends GraphNode, Link extends GraphLink<Node>> {
  nodes: Node[];
  links: Link[];
}

export interface ForceGraph3DOptions<Node extends GraphNode, Link extends GraphLink<Node>> {
  center?: readonly [x: number, y: number, z: number];
  chargeStrength?: number | ((node: Node, index: number) => number);
  collisionRadius?: number | ((node: Node, index: number) => number);
  collisionStrength?: number;
  linkDistance?: number | ((link: Link, index: number) => number);
  linkStrength?: number | ((link: Link, index: number) => number);
  velocityDecay?: number;
  onTick?: (graph: Readonly<MutableGraph<Node, Link>>) => void;
  onEnd?: (graph: Readonly<MutableGraph<Node, Link>>) => void;
}

export interface ForceGraph3D<Node extends GraphNode, Link extends GraphLink<Node>> {
  readonly graph: Readonly<MutableGraph<Node, Link>>;
  readonly simulation: Simulation<Node>;
  setData(data: GraphData<Node, Link>): void;
  reheat(alpha?: number): void;
  settle(iterations?: number): Readonly<MutableGraph<Node, Link>>;
  stop(): void;
}

const copyGraph = <Node extends GraphNode, Link extends GraphLink<Node>>(
  data: GraphData<Node, Link>,
): MutableGraph<Node, Link> => ({
  // d3 mutates positions and replaces link ids with node references.
  nodes: data.nodes.map((node) => ({ ...node })),
  links: data.links.map((link) => ({ ...link })),
});

/** Create a renderer-independent 3D layout for use with WebGL, Three.js, or Canvas. */
export function createForceGraph3D<Node extends GraphNode, Link extends GraphLink<Node>>(
  data: GraphData<Node, Link>,
  options: ForceGraph3DOptions<Node, Link> = {},
): ForceGraph3D<Node, Link> {
  let graph = copyGraph(data);
  const center = options.center ?? [0, 0, 0];
  const links = forceLink<Node, Link>(graph.links)
    .id((node) => node.id)
    .distance(options.linkDistance ?? 30)
    .strength(options.linkStrength ?? 1);

  const simulation = forceSimulation<Node>(graph.nodes, 3)
    .numDimensions(3)
    .force("link", links)
    .force("charge", forceManyBody<Node>().strength(options.chargeStrength ?? -80))
    .force("center", forceCenter<Node>(...center))
    .force("collision", forceCollide<Node>()
      .radius(options.collisionRadius ?? 4)
      .strength(options.collisionStrength ?? 0.7))
    .velocityDecay(options.velocityDecay ?? 0.4);

  const api: ForceGraph3D<Node, Link> = {
    get graph() { return graph; },
    simulation,
    setData(nextData) {
      graph = copyGraph(nextData);
      simulation.nodes(graph.nodes);
      links.links(graph.links);
      api.reheat();
    },
    reheat(alpha = 1) { simulation.alpha(alpha).restart(); },
    settle(iterations = 300) {
      simulation.stop().tick(iterations);
      options.onTick?.(graph);
      return graph;
    },
    stop() { simulation.stop(); },
  };

  simulation
    .on("tick", () => options.onTick?.(graph))
    .on("end", () => options.onEnd?.(graph));

  return api;
}

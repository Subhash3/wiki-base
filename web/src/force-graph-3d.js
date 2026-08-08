import {
  forceCenter,
  forceCollide,
  forceLink,
  forceManyBody,
  forceSimulation,
} from "d3-force-3d";

const copyGraph = (data) => ({
  nodes: data.nodes.map((node) => ({ ...node })),
  links: data.links.map((link) => ({ ...link })),
});

/** Browser build of the typed force-graph-3d module. */
export function createForceGraph3D(data, options = {}) {
  let graph = copyGraph(data);
  const center = options.center ?? [0, 0, 0];
  const links = forceLink(graph.links)
    .id((node) => node.id)
    .distance(options.linkDistance ?? 30)
    .strength(options.linkStrength ?? 1);
  const simulation = forceSimulation(graph.nodes, 3)
    .numDimensions(3)
    .force("link", links)
    .force("charge", forceManyBody().strength(options.chargeStrength ?? -80))
    .force("center", forceCenter(...center))
    .force("collision", forceCollide()
      .radius(options.collisionRadius ?? 4)
      .strength(options.collisionStrength ?? 0.7))
    .velocityDecay(options.velocityDecay ?? 0.4);

  const api = {
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

/** The package does not publish TypeScript declarations; this describes the API used here. */
declare module "d3-force-3d" {
  export interface SimulationNodeDatum {
    index?: number;
    x?: number; y?: number; z?: number;
    vx?: number; vy?: number; vz?: number;
    fx?: number | null; fy?: number | null; fz?: number | null;
  }

  export interface SimulationLinkDatum<Node extends SimulationNodeDatum> {
    source: Node | string | number;
    target: Node | string | number;
    index?: number;
  }

  export interface Force<Node extends SimulationNodeDatum> {
    (alpha: number): void;
    initialize?(nodes: Node[], random: () => number, dimensions: number): void;
  }

  export interface Simulation<Node extends SimulationNodeDatum> {
    numDimensions(): number;
    numDimensions(dimensions: 1 | 2 | 3): this;
    nodes(): Node[];
    nodes(nodes: Node[]): this;
    alpha(): number;
    alpha(alpha: number): this;
    alphaTarget(): number;
    alphaTarget(target: number): this;
    force(name: string, force: Force<Node> | null): this;
    restart(): this;
    stop(): this;
    tick(iterations?: number): this;
    on(type: "tick" | "end", listener: (() => void) | null): this;
    velocityDecay(decay: number): this;
  }

  export interface LinkForce<Node extends SimulationNodeDatum, Link extends SimulationLinkDatum<Node>> extends Force<Node> {
    id(id: (node: Node) => string | number): this;
    links(links: Link[]): this;
    distance(distance: number | ((link: Link, index: number) => number)): this;
    strength(strength: number | ((link: Link, index: number) => number)): this;
  }

  export interface ManyBodyForce<Node extends SimulationNodeDatum> extends Force<Node> {
    strength(strength: number | ((node: Node, index: number) => number)): this;
  }

  export interface CollideForce<Node extends SimulationNodeDatum> extends Force<Node> {
    radius(radius: number | ((node: Node, index: number) => number)): this;
    strength(strength: number): this;
  }

  export function forceSimulation<Node extends SimulationNodeDatum>(nodes?: Node[], dimensions?: 1 | 2 | 3): Simulation<Node>;
  export function forceLink<Node extends SimulationNodeDatum, Link extends SimulationLinkDatum<Node>>(links?: Link[]): LinkForce<Node, Link>;
  export function forceManyBody<Node extends SimulationNodeDatum>(): ManyBodyForce<Node>;
  export function forceCenter<Node extends SimulationNodeDatum>(x?: number, y?: number, z?: number): Force<Node>;
  export function forceCollide<Node extends SimulationNodeDatum>(): CollideForce<Node>;
}

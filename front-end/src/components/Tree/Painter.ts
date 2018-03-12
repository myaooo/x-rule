import * as d3 from 'd3';

import { TreeNode, isInternalNode, InternalNode } from '../../models';
import * as utils from '../../service/utils';
import * as nt from '../../service/num';
import { Painter, ColorType, labelColor, defaultDuration } from '../Painters';

interface D3Node extends d3.HierarchyNode<TreeNode> {
  _children?: D3Node[];
}
interface D3PointNode extends d3.HierarchyPointNode<TreeNode> {
  x0?: number;
  y0?: number;
  _children?: D3PointNode[];
  // idx?: number;
  nth?: number;
  expanded?: boolean;
  tspan?: string;
  title?: string;
}

interface Point {
  x: number;
  y: number;
}

export function createHierarchy(treeRoot: TreeNode, threshold: number = 0.01): D3Node {
  return d3.hierarchy<TreeNode>(treeRoot, (node: TreeNode) => {
    return isInternalNode(node) ? [node.left, node.right] : [];
  });
}

interface NodePainterOptionalParams {
  fontSize: number;
  duration: number;
  nodeSize: [number, number];
  color: ColorType;
  featureName(i: number): string;
}

interface NodePainterParams extends Partial<NodePainterOptionalParams> {
  source: D3PointNode;
  onClick(d: D3PointNode): void;
}

class NodePainter implements Painter<D3PointNode[], NodePainterParams> {
  public static defaultProps: NodePainterOptionalParams = {
    fontSize: 12,
    duration: 400,
    nodeSize: [120, 180],
    color: labelColor,
    featureName: (i: number): string => `X${i}`,
  };
  private nodes: D3PointNode[];
  private params: NodePainterParams & NodePainterOptionalParams;

  public data(nodes: D3PointNode[]) {
    this.nodes = nodes;
    return this;
  }
  public update(
    params: NodePainterParams
  ) {
    this.params = {...NodePainter.defaultProps, ...this.params, ...params};

    return this;
  }

  public render(
    selector: d3.Selection<SVGElement, any, any, any>, 
  ) {
    const node = this.doJoin(selector);
    const nodeEnter = this.doEnter(node.enter());
    this.doUpdate(nodeEnter.merge(node));
    this.doExit(node.exit());
    return this;
  }

  public doJoin(
    selector: d3.Selection<SVGElement, D3PointNode, any, any>
  ): d3.Selection<SVGGElement, D3PointNode, any, any> {
    selector
      .selectAll('g.nodes')
      .data([''])
      .enter()
      .append('g')
      .attr('class', 'nodes');
    const nodeGroup = d3.select('g.nodes');

    const node = nodeGroup
      .selectAll<SVGGElement, D3PointNode>('g.node')
      .data(this.nodes, (d: D3PointNode) => String(d.data.idx));
    return node;
  }

  public doEnter(
    nodeEnter: d3.Selection<d3.EnterElement, D3PointNode, any, any>
  ): d3.Selection<SVGGElement, D3PointNode, any, any> {
    const { source, color } = this.params;
    // const {fontSize, onClick} = this.params;
    // Add new node
    const nodeEntered = nodeEnter
      .append<SVGGElement>('g')
      .attr('class', 'node')
      .attr('transform', (d: D3PointNode) => `translate(${source.x0},${source.y0})`);

    // Add rect box for the nodes
    nodeEntered
      .append('rect')
      .attr('class', 'node')
      .attr('width', 1e-6)
      .attr('height', 1e-6)
      // .attr('rx', 1e-6)
      .style('stroke', (d: D3PointNode) => color(d.data.output))
      .style('fill', (d: D3PointNode) => isInternalNode(d.data) ? '#ddd' : '#fff');

    // Add labels for the nodes
    const text = nodeEntered
      .append('text')
      .attr('dy', '.35em')
      // .attr('x', 0)
      // .attr('y', fontSize)
      .attr('text-anchor', 'middle');
    text.append('tspan');
    text.append('title');
    return nodeEntered;
  }

  public doUpdate(nodeUpdate: d3.Selection<SVGGElement, D3PointNode, any, any>): void {
    const { duration, nodeSize, featureName, fontSize, onClick } = this.params;
    const nodeWidth = nodeSize[0];
    const nodeHeight = nodeSize[1];
    // Transition to the proper position for the node
    nodeUpdate
      .transition()
      .duration(duration)
      .attr('transform', (d: D3PointNode) => `translate(${d.x},${d.y - nodeHeight})`);

    // Update the node attributes and style
    nodeUpdate
      .select('rect.node')
      .attr('x', -nodeWidth / 2)
      .attr('rx', 2)
      .attr('ry', 2)
      .attr('width', nodeWidth)
      .attr('height', nodeHeight)
      // .style('fill', (d: D3PointNode) => {
      //   return isInternalNode(d.data) ? 'lightsteelblue' : '#fff';
      // })
      .attr('cursor', 'pointer')
      .on('click', onClick);

    this.nodes.forEach((d: D3PointNode): any => {
      const rank = d.nth;
      if (rank === undefined) return;
      if (rank < -1) {
        console.log('Error: mal tree format!'); // tslint:disable-line
      }
      if (rank < 0) {
        d.tspan = '', d.title = '';
        return;
      }
      const parent = <InternalNode> (<D3PointNode> d.parent).data;
      // const feature = parent.feature;
      const tmp: (number | null)[] = [null, null];
      if (rank === 0) tmp[1] = parent.threshold;
      else tmp[0] = parent.threshold;
      console.log(parent); // tslint:disable-line
      const { tspan, title } = utils.condition2String(featureName(parent.feature), tmp);
      d.tspan = tspan;
      d.title = title;
    });
    // Update texts
    const nodeText = nodeUpdate
      .select('text')
      .attr('x', 0)
      .attr('y', fontSize);

    nodeText.select('tspan').text((d: D3PointNode, i: number) => d.tspan || '');
    nodeText.select('title').text((d: D3PointNode, i: number) => d.title || '');
  }

  public doExit(nodeExit: d3.Selection<Element, D3PointNode, any, any>): void {
    const { source, duration } = this.params;
    const nodeExited = nodeExit
      .transition()
      .duration(duration || defaultDuration)
      .attr('transform', (d: D3PointNode) => `translate(${source.x},${source.y})`)
      .remove();

    // On exit reduce the node circles size to 0
    nodeExited
      .select('rect')
      .attr('width', 1e-6)
      .attr('height', 1e-6);

    // On exit reduce the opacity of text labels
    nodeExited.select('text').style('fill-opacity', 1e-6);
  }
}

interface LinkPainterOptionalParams {
  duration: number;
  nodeSize: [number, number];
  margin: { bottom: number };
  linkWidthMultiplier: number;
  color: ColorType;
}

interface LinkPainterParams extends Partial<LinkPainterOptionalParams> {
  source: D3PointNode;
}

class LinkPainter implements Painter<D3PointNode[], LinkPainterParams> {
  public static defaultParams = {
    duration: 400,
    nodeSize: [120, 180],
    margin: {bottom: 50},
    color: labelColor,
    linkWidthMultiplier: 1,
  };
  private links: D3PointNode[];
  private params: LinkPainterOptionalParams & LinkPainterParams;
  // private source: D3PointNode;
  // private maxSupportSum: number;
  // constructor() {
  // }
  public data(links: D3PointNode[]) {
    this.links = links;
    return this;
  }
  public render(
    selector: d3.Selection<SVGElement, any, any, any>, 
  ) {
    const link = this.doJoin(selector);
    const linkEnter = this.doEnter(link.enter());
    this.doUpdate(linkEnter.merge(link));
    this.doExit(link.exit());
    return this;
  }
  public update(
    params: LinkPainterParams
  ) {
    this.params = {...(LinkPainter.defaultParams), ...(this.params), ...params};
    return this;
  }

  public doJoin(
    selector: d3.Selection<SVGElement, D3PointNode, any, any>
  ): d3.Selection<SVGGElement, D3PointNode, any, any> {
    // make sure there is a nodes group exists;
    selector
      .selectAll('g.links')
      .data(['links'])
      .enter()
      .append('g')
      .attr('class', 'links');
    const linkGroup = d3.select('g.links');

    const link = linkGroup
      .selectAll<SVGGElement, D3PointNode>('g.link')
      .data<D3PointNode>(this.links, (d: D3PointNode) => String(d.data.idx));
    return link;
  }

  public doEnter(
    entered: d3.Selection<d3.EnterElement, D3PointNode, any, any>
  ): d3.Selection<SVGGElement, D3PointNode, any, any> {
    const { margin , source} = this.params;
    // Enter any new links at the parent's previous position.
    const linkEnter = entered.append<SVGGElement>('g').attr('class', 'link');
    const links = linkEnter.selectAll('path.link')
      .data((d: D3PointNode) => d.data.value);
    //
    links.enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', (d: number) => {
        const o = { x: source.x0 || 0, y: (source.y0 || 0) + margin.bottom };
        return diagonal(o, o);
      });

    return linkEnter;
  }

  public doUpdate(linkUpdate: d3.Selection<SVGGElement, D3PointNode, any, any>): void {
    type Support = {sum: number; value: number, acc: number, node: D3PointNode};
    const { duration, nodeSize, linkWidthMultiplier, color } = this.params;
    const multiplier = linkWidthMultiplier;
    const nodeHeight = nodeSize[1];
    // Transition back to the parent element position
    const links = linkUpdate
      .selectAll('path.link')
      .data((d: D3PointNode): Support[] => {
        const sum = nt.sum(d.data.value);
        let accumulate = 0;
        return d.data.value.map((v: number) => {
          accumulate += v;
          return {
            sum: sum * multiplier,
            value: v * multiplier,
            acc: (accumulate - v) * multiplier,
            node: d
          };
        });
      });
    links
      .transition()
      .duration(duration)
      .attr('d', (d: Support) => {
        const node = d.node;
        const parent = node.parent;
        if (!parent) return '';
        const s = { x: node.x + d.acc - d.sum / 2, y: node.y - nodeHeight };
        const t = { x: parent.x + d.acc - (node.nth === 0 ? d.sum : 0), y: parent.y };
        // const t = { x: (d.parent ? d.parent.x : 0), y: (d.parent ? d.parent.y + margin.top : 0)};
        return diagonal(s, t, d.value);
      })
      .style('fill', (d: Support, i: number) => color(i));
  }

  public doExit(linkExit: d3.Selection<Element, D3PointNode, any, any>): void {
    const { duration, source } = this.params;
    // Remove any exiting links
    linkExit
      .transition()
      .duration(duration)
      .style('fill-opacity', 1e-6)
      .remove();
    linkExit.selectAll('path.link')
      .data((d: D3PointNode) => d.data.value)
      .transition()
      .duration(duration)
      .attr('d', (d: number) => {
        const o = { x: source.x, y: source.y };
        return diagonal(o, o);
      });
  }
}

interface OptionalTreeStyles {
  nodeSize: [number, number]; // [width, height]
  margin: { top: number; right: number; bottom: number; left: number };
  duration: number;
  fontSize: number;
  color: ColorType;
  linkWidth: number;
}

export interface DrawTreeParams extends Partial<OptionalTreeStyles> {
  height: number;
  width: number;
  onClick?: (d: D3PointNode) => void;
  featureName: (idx: number) => string;
}

export class TreePainter implements Painter<D3Node, DrawTreeParams> {
  public static defaultParams: OptionalTreeStyles = {
    nodeSize: [80, 60],
    margin: { top: 5, right: 10, left: 10, bottom: 30 },
    duration: 400,
    fontSize: 12,
    color: labelColor,
    linkWidth: 1,
  };
  private treeMap: d3.TreeLayout<TreeNode>;
  private hierarchyRoot: D3Node;
  // private styles: DrawTreeParams & OptionalTreeStyles;
  private params: DrawTreeParams & OptionalTreeStyles;
  // private onClick: ()
  // private nodeSize: [number, number];
  private nodePainter: NodePainter;
  private linkPainter: LinkPainter;
  constructor(treeData: TreeNode) {
    this.data(createHierarchy(treeData));
  }
  public data(root: D3Node) {
    this.hierarchyRoot = root;
    this.collapse();
    return this;
  }
  public render(
    selector: d3.Selection<SVGElement, any, d3.BaseType, any>,
  ) {
    const { width, margin, nodeSize } = this.params;
    // const nodeWidth = nodeSize[0] - margin.left - margin.right;
    // const nodeHeight = nodeSize[1] - margin.top - margin.bottom;
    const nodeBox = [
      nodeSize[0] + margin.left + margin.right, 
      nodeSize[1] + margin.top + margin.bottom
    ] as [number, number];
    this.treeMap = d3
      .tree<TreeNode>()
      .separation((a, b) => (a.parent === b.parent ? 1 : 1.2))
      .nodeSize(nodeBox);
    const rootInit = { ...this.hierarchyRoot, x0: width / 2, y0: 0, x: width / 2, y: 0 } as D3PointNode;
    console.log('update from root'); //tslint:disable-line
    this._render(selector, rootInit);
    return this;
  }
  public update(
    params: DrawTreeParams,
  ) {
    this.params = {...(TreePainter.defaultParams), ...(this.params), ...params };
    return this;
  }

  public collapse() {
    const root = this.hierarchyRoot;
    root.descendants().forEach((node: D3PointNode) => {
      if (node.data.collapsed && node.children) {
        node._children = node.children;
        node.children = undefined;
      }
    });
    return this;
  }

  private _render(selector: d3.Selection<SVGElement, any, d3.BaseType, any>, source: D3PointNode): void {
    const { onClick, featureName } = this.params;
    const { duration, margin, nodeSize, fontSize, width, color, linkWidth } = this.params;
    const nodeDepth = nodeSize[1] + margin.top + margin.bottom;
    selector.attr('transform', `translate(${width / 2}, ${nodeSize[1] + margin.top})`);

    const root = this.treeMap(this.hierarchyRoot);
    const nodes = root.descendants();
    // console.log("#nodes:" + nodes.length); // tslint:disable-line
    nodes.forEach((d: D3PointNode) => {
      // d.idx = d.idx || i++;
      d.nth = d.nth || nthChild(d);
    });
    const links = nodes.slice(1);

    nodes.forEach((d: D3PointNode) => (d.y = d.depth * nodeDepth));

    const click = (d: D3PointNode) => {
      console.log('clicked'); // tslint:disable-line
      console.log(d); // tslint:disable-line
      if (d.children) {
        d._children = d.children;
        d.children = undefined;
      } else if (d._children) {
        d.children = d._children;
        d._children = undefined;
      }
      if (onClick) onClick(d);
      this._render(selector, d);
    };
    const linkParams = { 
      duration, 
      margin, 
      color,
      source,
      nodeSize: nodeSize, 
      linkWidthMultiplier: nodeSize[0] * linkWidth / 2 / nt.sum(nodes[0].data.value),
    };
    if (!this.linkPainter) this.linkPainter = new LinkPainter();
    this.linkPainter.update(linkParams).data(links).render(selector);

    const nodeParams = {
      fontSize,
      duration,
      featureName,
      nodeSize,
      source,
      onClick: click
    };
    if (!this.nodePainter) this.nodePainter = new NodePainter();
    this.nodePainter.update(nodeParams).data(nodes).render(selector);
    // new NodePainter(nodes).update();
    // Store the old positions for transition.
    nodes.forEach((d: any) => {
      d.x0 = d.x;
      d.y0 = d.y;
    });
  }
}

// export function drawTree(
//   selector: d3.Selection<SVGElement, any, d3.BaseType, any>,
//   treeData: TreeNode,
//   params: DrawTreeParams,
//   styles: DrawTreeStyles
// ) {
//   return new TreePainter(selector, treeData, params, styles);
// }

function nthChild(node: D3PointNode): number {
  const parent = node.parent;
  if (parent === null) return -1;
  const siblings = parent.children;
  if (siblings === undefined) return -2;
  for (let i = 0; i < siblings.length; ++i) {
    if (siblings[i] === node) return i;
  }
  return -3;
}

// Creates a curved (diagonal) path from parent to the child nodes
function diagonal(s: Point, d: Point, w: number = 0) {
  const midY = (s.y + d.y) / 2;
  const path = `M ${s.x} ${s.y}
                C ${s.x} ${midY}, ${d.x} ${midY},${d.x} ${d.y}
                h ${w}
                C ${d.x + w} ${midY}, ${s.x + w} ${midY},${s.x + w} ${s.y}
                Z`;

  return path;
}

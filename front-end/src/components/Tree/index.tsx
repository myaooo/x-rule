import * as React from 'react';
import * as d3 from 'd3';
import './index.css';

interface TreeData {
    text: string;
    children?: TreeData[];
}

class TreeNode {
    children: TreeNode[];
    text: string;
    parent: TreeNode | null;
    id: number;
    constructor(data: { text: string }, parent: TreeNode | null = null) {
        this.text = data.text;
        this.children = [];
        this.parent = parent;
    }
    pushChild(node: TreeNode) {
        this.children.push(node);
    }
    hasChild() {
        return this.children.length === 0;
    }
    hasParent() {
        return this.parent !== null;
    }
}

type D3PointNode = d3.HierarchyPointNode<TreeNode>;
interface PointNode extends D3PointNode {
    _children?: PointNode[];
    x0?: number;
    y0?: number;
}
interface Point {
    x: number;
    y: number;
}

// function buildTree(treeData: { root: string; nodes: any }): TreeNode {
//     const nodes = treeData.nodes;
//     const rootIdx = treeData.root;
//     function _buildTree(idx: string) {
//         const node = new TreeNode(nodes[idx]);
//         const childrenIndices = nodes[idx].children;
//         childrenIndices.every((childIdx: string) => {
//             node.pushChild(_buildTree(childIdx));
//         });
//         return node;
//     }
//     return _buildTree(rootIdx);
// }

function buildTree(treeData: TreeData, parent: TreeNode | null = null): TreeNode {
    const node = new TreeNode(treeData, parent);
    if (treeData.children)
        treeData.children.forEach((child) => node.pushChild(buildTree(child, node)));
    return node;
}

export interface TreeViewProps {
    duration?: number;
    width?: number;
    height?: number;
    margin?: {top: number, right: number, bottom: number, left: number};
    treeData: TreeData;
}

interface TreeViewState {
    treeData: TreeData;
    width: number;
    height: number;
    margin: {top: number, right: number, bottom: number, left: number};
}
// type D3Node = d3.HierarchyNode<TreeNode>;

class TreeView extends React.Component<TreeViewProps, TreeViewState> {
    ref: SVGSVGElement;
    g: d3.Selection<Element, any, any, any>;
    duration: number;
    rootNode: PointNode | null;
    width: number;
    height: number;

    // root: D3Node;
    constructor(props: TreeViewProps) {
        super(props);
        this.duration = props.duration ? props.duration : 700;
        const margin = props.margin ? props.margin : {top: 20, right: 90, bottom: 30, left: 90};
        this.state = { 
            treeData: props.treeData,
            width: props.width ? props.width : 720,
            height: props.height ? props.height : 480,
            margin,
        };
        this.width = this.state.width - margin.left - margin.right,
        this.height = this.state.height - margin.top - margin.bottom;
        this.rootNode = null;
        this.update = this.update.bind(this);
    }
    render() {
        return (
            <svg 
                width={this.state.width} 
                height={this.state.height} 
            >
            <g 
                transform={`translate(${this.state.margin.left},${this.state.margin.top})`}
                ref={(g: SVGSVGElement) => { this.ref = g; }} 
            />
            </svg>
        );
    }
    componentDidMount() {
        this.g = d3.select(this.ref);
        this.drawTree();
        // this.svg = d3.select('#Tree');
    }
    drawTree(treeData: TreeData = this.state.treeData) {
        const root = buildTree(treeData);
        const treeMap = d3.tree<TreeNode>().size([this.height, this.width]);
        // this.root = d3.hierarchy(root, (node) => node.children);
        this.rootNode = treeMap(d3.hierarchy(root, (node) => node.children)) as PointNode;
        this.rootNode.x0 = this.height / 2;
        this.rootNode.y0 = 0;
        this.update(this.rootNode);

    }
    update(source: PointNode) {

        const hasChildren = (children: PointNode[] | undefined) => children && children.length > 0;

        // Toggle children on click.
        const click = (d: PointNode) => {
            if (hasChildren(d.children)) {
                d._children = d.children;
                d.children = [];
            } else {
                d.children = d._children;
                d._children = [];
            }
            this.update(d);
        };
        if (this.rootNode === null) return;
        let i = 0;

        // Assigns the x and y position for the nodes

        // Compute the new tree layout.
        const nodes = this.rootNode.descendants();
        const links = nodes.slice(1);

        // Normalize for fixed-depth.
        nodes.forEach((d: PointNode) => { d.y = d.depth * 180; });

        // ****************** Nodes section ***************************

        // Update the nodes...
        const node = this.g.selectAll('g.node')
            .data<PointNode>(nodes, (d: any) => { return d.id || (d.id = ++i); });

        // Enter any new modes at the parent's previous position.
        const nodeEnter = node.enter().append('g')
            .attr('class', 'node')
            .attr('transform', (d: PointNode) => {
                return 'translate(' + source.y0 + ',' + source.x0 + ')';
            })
            .on('click', click);

        // Add Circle for the nodes
        nodeEnter.append('circle')
            .attr('class', 'node')
            .attr('r', 1e-6)
            .style('fill', (d: PointNode) => {
                return hasChildren(d._children) ? 'lightsteelblue' : '#fff';
            });

        // Add labels for the nodes
        nodeEnter.append('text')
            .attr('dy', '.35em')
            .attr('x', (d: PointNode) => {
                return d.children || d._children ? -13 : 13;
            })
            .attr('text-anchor', (d: PointNode) => {
                return d.children || d._children ? 'end' : 'start';
            })
            .text((d) => { return d.data.text; });

        // UPDATE
        const nodeUpdate = nodeEnter.merge(node);

        // Transition to the proper position for the node
        nodeUpdate.transition()
            .duration(this.duration)
            .attr('transform', (d: PointNode) => {
                return 'translate(' + d.y + ',' + d.x + ')';
            });

        // Update the node attributes and style
        nodeUpdate.select('circle.node')
            .attr('r', 10)
            .style('fill', (d: PointNode) => {
                return hasChildren(d._children) ? 'lightsteelblue' : '#fff';
            })
            .attr('cursor', 'pointer');

        // Remove any exiting nodes
        const nodeExit = node.exit().transition()
            .duration(this.duration)
            .attr('transform', (d: PointNode) => {
                return 'translate(' + source.y + ',' + source.x + ')';
            })
            .remove();

        // On exit reduce the node circles size to 0
        nodeExit.select('circle')
            .attr('r', 1e-6);

        // On exit reduce the opacity of text labels
        nodeExit.select('text')
            .style('fill-opacity', 1e-6);

        // ****************** links section ***************************

        // Update the links...
        const link = this.g.selectAll('path.link')
            .data(links, (d: any) => { return d.id; });

        // Enter any new links at the parent's previous position.
        const linkEnter = link.enter().insert('path', 'g')
            .attr('class', 'link')
            .attr('d', (d: PointNode) => {
                const o = { x: source.x0 || 0, y: source.y0 || 0 };
                return diagonal(o, o);
            });

        // UPDATE
        const linkUpdate = linkEnter.merge(link);

        // Transition back to the parent element position
        linkUpdate.transition()
            .duration(this.duration)
            .attr('d', (d: PointNode) => { return d.parent ? diagonal(d, d.parent) : 0; });

        // Remove any exiting links
        link.exit().transition()
            .duration(this.duration)
            .attr('d', (d: PointNode) => {
                const o = { x: source.x, y: source.y };
                return diagonal(o, o);
            })
            .remove();

        // Store the old positions for transition.
        nodes.forEach((d: any) => {
            d.x0 = d.x;
            d.y0 = d.y;
        });

        // Creates a curved (diagonal) path from parent to the child nodes
        function diagonal(s: Point, d: Point) {

            const path = `M ${s.y} ${s.x}
                  C ${(s.y + d.y) / 2} ${s.x},
                    ${(s.y + d.y) / 2} ${d.x},
                    ${d.y} ${d.x}`;

            return path;
        }

    }

}

export default TreeView;
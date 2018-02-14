import * as React from 'react';
import * as d3 from 'd3';

import { TreeModel, PlainData } from '../../models';
import './index.css';
import { TreePainter, createHierarchy } from './Painter';
import { TreeStyles } from '../../store';

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

// interface Margin {
//   top: number; right: number; bottom: number; left: number;
// }

interface OptionalProps {
  duration: number;
  width: number;
  height: number;
  // margin: Margin;
  transform: string;
}

export interface TreeViewProps extends Partial<OptionalProps> {
  styles?: TreeStyles;
  model: TreeModel;
  data: (PlainData | undefined)[];
}

interface TreeViewState {
}
// type D3Node = d3.HierarchyNode<TreeNode>;

export default class TreeView extends React.Component<TreeViewProps, TreeViewState> {

  public static defaultProps: OptionalProps = {
    duration: 400,
    width: 900,
    height: 700,
    // margin: { top: 20, right: 90, bottom: 30, left: 90 },
    transform: '',
  };
  private ref: SVGSVGElement;
  private needUpdate: boolean;
  private painter: TreePainter;
  // private g: d3.Selection<Element, any, any, any>;
  // private rootNode: PointNode | null;
  // root: D3Node;
  constructor(props: TreeViewProps) {
    super(props);
    this.painter = new TreePainter(props.model.root);
    // this.rootNode = null;
  }

  public render() {
    // const { width, height } = this.props;
    const transform = this.props.transform;
    return (
      // <svg width={width} height={height}>
      <g
        transform={transform}
        ref={(g: SVGSVGElement) => {
          this.ref = g;
        }}
      />
      // </svg>
    );
  }
  componentDidMount() {
    this.update();
    // this.svg = d3.select('#Tree');
  }
  componentWillReceiveProps(nextProps: TreeViewProps) {
    // console.log('Entering will receive props');  // tslint:disable-line
    if (nextProps.data.length !== this.props.data.length) {
      // console.log('Update flag set to true');  // tslint:disable-line
      this.needUpdate = true;
    }
    if (nextProps.model !== this.props.model) {
      const root = createHierarchy(nextProps.model.root);
      
      this.painter.data(root);
      this.needUpdate = true;
    }
    if (nextProps.styles !== this.props.styles) {
      this.needUpdate = true;
    }
  }
  shouldComponentUpdate(nextProps: TreeViewProps, nextState: TreeViewState) {
    // console.log('Entering should component update');  // tslint:disable-line
    return Boolean(this.needUpdate);
  }
  componentDidUpdate() {
    console.log('Updating');  // tslint:disable-line
    this.update();
    this.needUpdate = false;
  }
  private update() {
    const { data, width, height, styles } = this.props as TreeViewProps & OptionalProps;
    const availableData = data[0] || data[1];
    const featureName = availableData 
      ? ((i: number) => availableData.featureNames[i]) 
      : ((i: number) => `X${i}`);
    const params = { ...styles, width, height, featureName };
    this.painter.update(params).render(d3.select(this.ref));
  }

}

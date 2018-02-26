import * as React from "react";
import {
  Transition,
  PlainObject,
} from '../core';

export interface INodeGroupProps<D, S> {
  /**
   * An array.  The data prop is treated as immutable so the nodes will only update if prev.data !== next.data.
   */
  data: Array<D>;
  /**
   * Function that returns a string key given a data object and its index.  Used to track which nodes are entering, updating and leaving.
   */
  keyAccessor: (data: D, index: number) => string;
  /**
  * A function that returns the starting state.  The function is passed the data and index and must return an object.
  */
  start: (data: D, index: number) => S;
  /**
   * A function that **returns an object or array of objects** describing how the state should transform on enter.  The function is passed the data and index.
   */
  enter?: (data: D, index: number) => Transition | Array<Transition>;
  /**
   * A function that **returns an object or array of objects** describing how the state should transform on update.  The function is passed the data and index.
   */
  update?: (data: D, index: number) => Transition | Array<Transition>;
  /**
   * A function that **returns an object or array of objects** describing how the state should transform on leave.  The function is passed the data and index.
   */
  leave?: (data: D, index: number) => Transition | Array<Transition>;
  /**
   * A function that renders the nodes. It should accept an array of nodes as its only argument.  Each node is an object with the key, data, state and a type of 'ENTER', 'UPDATE' or 'LEAVE'.
   */
  children: <C>(nodes: Array<{key: string; data: D; state: S}>) => React.ReactElement<C>;
}

declare class NodeGroup<DataType = {}, State = {}> extends React.Component<INodeGroupProps<DataType, State>> { }

export default NodeGroup;

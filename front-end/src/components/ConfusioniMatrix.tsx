import * as React from 'react';
import { Painter } from './Painters/Painter';
import * as d3 from 'd3';

interface ConfusionParams {
  width: number;
  height: number;
}

type ConfusionData = number[][];

class ConfusionPainter implements Painter<ConfusionData, ConfusionParams> {
  private matrix: ConfusionData;
  private params: ConfusionParams;
  update(params: ConfusionParams): this {
    this.params = {...(this.params), ...params};
    return this;
  }
  data(newData: ConfusionData): this {
    this.matrix = newData;
    return this;
  }
  render<GElement extends d3.BaseType>(selector: d3.Selection<SVGElement, ConfusionData, GElement, any>): this {
    // Render grid
    const grid = selector.select<SVGGElement>('g.grid');
    this.renderGrid(grid);
    return this;
  }

  renderGrid(selector: d3.Selection<SVGGElement, ConfusionData, any, any>): this {
    const {width, height} = this.params;
    const nClass = this.matrix.length;
    const stepX = width / nClass;
    const stepY = height / nClass;
    // Render grid
    const grids = d3.range(nClass + 1);
    const gridX = selector.selectAll('path.grid-x').data(grids);
    const gridXUpdate = gridX.enter().append('path').attr('class', 'grid-x')
      .merge(gridX);
    gridXUpdate.attr('d', (d, i) => `M 0 ${i * stepY} H ${width}`);
    gridX.exit().transition().remove();

    const gridY = selector.selectAll('path.grid-y').data(grids);
    const gridYUpdate = gridX.enter().append('path').attr('class', 'grid-y')
      .merge(gridX);
    gridYUpdate.attr('d', (d, i) => `M ${i * stepX} 0 V ${height}`);
    gridY.exit().transition().remove();

    return this;
  }

  renderRects(selector: d3.Selection<SVGGElement, ConfusionData, any, any>): this {

    return this;
  }
}

export interface ConfusionMatrixProps {
  confusion: number[][];
}

export interface ConfusionMatrixState {
}

export default class ConfusionMatrix extends React.Component<ConfusionMatrixProps, ConfusionMatrixState> {
  private painter: ConfusionPainter;
  private ref: SVGSVGElement;
  constructor(props: ConfusionMatrixProps) {
    super(props);
    this.painter = new ConfusionPainter();
  }
  update() {
    this.painter.render(d3.select(this.ref));
  }
  render() {
    return (
      <div>
        <svg id="confusion"/>  
      </div>
    );
  }
}

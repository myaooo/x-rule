import * as React from 'react';
import { Collapse, Button } from 'antd';
import { connect } from 'react-redux';
import { RootState, getModel } from '../store';
import DataTable from '../components/DataTable';
import dataService from '../service/dataService';
import { FilterType } from '../components/DataFilter';
import { getSelectedDataNames } from '../store/selectors';
import { ModelBase } from '../models';
import { DataTypeX } from '../models/data';
import { rankModelFeatures } from '../models/algos';

const Panel = Collapse.Panel;

export interface DataInputHeaderProps {
  onClick?: () => void;
}

export function DataInputHeader(props: DataInputHeaderProps) {
  const { onClick } = props;
  return (
    <div>
      Data Filter
      <Button onClick={onClick} type="primary" icon="upload" style={{ float: 'right', fontSize: 12 }} size="small">
        Filter
      </Button>
    </div>
  );
}

export interface DataFilterProps {
  model: ModelBase | null;
  // meta?: ModelMeta;
  // modelName: string;
  dataType?: DataTypeX;
}

export class DataFilter extends React.Component<DataFilterProps> {
  private tableRef: DataTable;
  constructor(props: DataFilterProps) {
    super(props);
    this.handleClickFilter = this.handleClickFilter.bind(this);
  }
  handleClickFilter() {
    return;
  }
  render() {
    const { model, dataType } = this.props;
    const width = 1200;
    const height = 150;
    const indices = model ? rankModelFeatures(model) : undefined;
    return (
      <Collapse defaultActiveKey={[]}>
        <Panel header={<DataInputHeader onClick={this.handleClickFilter}/>} key="1" style={{ width }}>
          {model && (
            <DataTable
              ref={(ref: DataTable) => this.tableRef = ref}
              meta={model.meta}
              height={height}
              getData={(filters: FilterType[], start?: number, end?: number) =>
                dataService.getFilterData(model.name, dataType, filters, start, end)
              }
              indices={indices}
            />
          )}
        </Panel>
      </Collapse>
    );
  }
}

const mapStateToProps = (state: RootState): DataFilterProps => {
  return {
    model: getModel(state),
    dataType: getSelectedDataNames(state)[0]
    // modelName
  };
};

export default connect(mapStateToProps)(DataFilter);

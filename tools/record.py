import redis
import json
import pandas as pd
import datetime as dt
from tqdm import tqdm  

def get_fills_from_redis(
    strat='capital_txo_main',
    acc='TAIFEX100',
    date=dt.date.today(),
    night_session=True,
    redishost='prod1.capital.radiant-knight.com'
):
    r = redis.StrictRedis(host=redishost, port=6379, db=0)
    rk = '{}:{}'.format(acc, date.strftime('%Y%m%d'))
    rk_strat = '{}:{}'.format(strat, date.strftime('%Y%m%d'))

    if night_session:
        rk = rk + 'E'
        rk_strat = rk_strat + 'E'

    keys = [k.decode() for k in r.keys()]
    if rk not in keys:
        None
        if rk_strat in keys:
            rk = rk_strat
        else:
            return None

    msgs = r.lrange(rk, 0, -1)
    trd_df = pd.DataFrame([json.loads(m.decode()) for m in msgs])

    if trd_df.empty:
        print(f'Empty redis key: {rk}')
        return None

    if trd_df['ts'][0] > 1e11:
        trd_df['time'] = trd_df['ts'].apply(lambda x: dt.datetime.fromtimestamp(x / 1e6))
    else:
        trd_df['time'] = trd_df['ts'].apply(lambda x: dt.datetime.fromtimestamp(x))

    trd_df.set_index('time', inplace=True)
    return trd_df



class RedisFillLoader:
    def __init__(self, start, end, strat="capital_electron_tmf", acc="TAIFEX100", redishost="prod1.capital.radiant-knight.com"):
        self.start = self._parse_date(start)
        self.end = self._parse_date(end)
        self.strat = strat
        self.acc = acc
        self.redishost = redishost
        self._realized_pnl = None
        self._net_position = None

    def _parse_date(self, date_input):
        if isinstance(date_input, str):
            parts = [int(x) for x in date_input.strip().split(",")]
            return dt.date(*parts)
        return date_input
    def _compute_pnl(self, df):
        from collections import defaultdict, deque

        realized_pnl = defaultdict(float)
        net_position = defaultdict(int)
        open_positions = defaultdict(deque)

        for _, row in df.iterrows():
            instr = row["instr"]
            px = row["px"]
            sz = row["sz"]
            mult = row["mult"]
            queue = open_positions[instr]
            realized = 0

            if sz * net_position[instr] >= 0 or not queue:
                queue.append({"px": px, "sz": sz, "mult": mult})
                net_position[instr] += sz
            else:
                remain = sz
                while remain != 0 and queue:
                    pos = queue[0]
                    matched_qty = -pos["sz"] if abs(remain) >= abs(pos["sz"]) else remain
                    trade_size = min(abs(remain), abs(pos["sz"]))
                    eff_mult = pos["mult"]
                    pnl = (px - pos["px"]) * trade_size * eff_mult
                    realized += pnl if pos["sz"] > 0 else -pnl

                    if abs(remain) >= abs(pos["sz"]):
                        remain += pos["sz"]
                        queue.popleft()
                    else:
                        pos["sz"] += remain
                        remain = 0

                if remain != 0:
                    queue.append({"px": px, "sz": remain, "mult": mult})
                net_position[instr] += sz

            realized_pnl[instr] += realized

        self._realized_pnl = dict(realized_pnl)
        self._net_position = dict(net_position)    
    def get_concat_df(self):
        dfs = []
        date_list = pd.date_range(start=self.start, end=self.end).to_pydatetime()

        for d in tqdm(date_list, desc="Concatenating fills"):
            d = d.date()
            for night in (True, False):
                df = get_fills_from_redis(
                    strat=self.strat,
                    acc=self.acc,
                    date=d,
                    night_session=night,
                    redishost=self.redishost
                )
                if df is not None and not df.empty:
                    dfs.append(df.reset_index())

        if not dfs:
            print("No data found in given date range.")
            return pd.DataFrame()

        df_all = pd.concat(dfs, ignore_index=True)
        df_all["time"] = pd.to_datetime(df_all["time"])
        df_all = df_all.sort_values("time").reset_index(drop=True)
        self._compute_pnl(df_all)
        return df_all

    def get_day_by_day_dict(self):
        df_dict = {}
        date_list = pd.date_range(start=self.start, end=self.end).to_pydatetime()

        for d in tqdm(date_list, desc="Loading fills by day"):
            d = d.date()
            dfs = []
            for night in (True, False):
                df = get_fills_from_redis(
                    strat=self.strat,
                    acc=self.acc,
                    date=d,
                    night_session=night,
                    redishost=self.redishost
                )
                if df is not None and not df.empty:
                    dfs.append(df.reset_index())

            if dfs:
                full = pd.concat(dfs, ignore_index=True)
                full["time"] = pd.to_datetime(full["time"])
                full = full.sort_values("time").reset_index(drop=True)
                df_dict[d] = full
            else:
                print(f"No data for {d}")

        return df_dict
    def get_realized(self):
        return self._realized_pnl

    def get_net_position(self):
        return self._net_position
    def get_pnl_summary(self) -> pd.DataFrame:
        import pandas as pd

        if self._realized_pnl is None or self._net_position is None:
            raise RuntimeError("請先呼叫 get_concat_df() 以產生損益資料")

        all_instr = sorted(set(self._realized_pnl) | set(self._net_position))
        rows = []
        for instr in all_instr:
            rows.append({
                "instr": instr,
                "realized_pnl": round(self._realized_pnl.get(instr, 0), 2),
                "net_position": self._net_position.get(instr, 0)
            })
        return pd.DataFrame(rows)
    def print_pnl_summary(self):
        from tabulate import tabulate

        if self._realized_pnl is None or self._net_position is None:
            raise RuntimeError("請先呼叫 get_concat_df() 以產生損益資料")

        all_instr = sorted(set(self._realized_pnl) | set(self._net_position))
        rows = []
        for instr in all_instr:
            rows.append([
                instr,
                round(self._realized_pnl.get(instr, 0), 2),
                self._net_position.get(instr, 0)
            ])

        headers = ["instr", "realized_pnl", "net_position"]
        print(tabulate(rows, headers=headers, tablefmt="plain"))

# 使用方法 => loader = RedisFillLoader("2025,5,1", "2025,5,6")
# df_all = loader.get_concat_df()
# df_dict = loader.get_day_by_day_dict()

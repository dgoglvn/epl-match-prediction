class Team:
    def __init__(self, name: str, att_rating: float, def_rating: float) -> None:
        self.name = name
        self.att_rating = att_rating
        self.def_rating = def_rating

    # def get_name(self) -> str:
    #     return self.name
    #
    # def set_name(self, name: str) -> None:
    #     for team in self.df["Team"]:
    #         if team == name:
    #             self.name = name
    #
    # def get_att_rating(self) -> float:
    #     return self.att_rating
    #
    # def set_att_rating(self, name: str) -> None:
    #     filtered_df = self.df.loc[self.df["Team"] == name]
    #     self.att_rating = filtered_df["ATT"].iloc[0]
    #
    # def get_def_rating(self) -> float:
    #     return self.def_rating
    #
    # def set_def_rating(self, name: str) -> None:
    #     filtered_df = self.df.loc[self.df["Team"] == name]
    #     self.def_rating = filtered_df["DEF"].iloc[0]

    def __repr__(self) -> str:
        return f"<Team {self.name}: ATT={self.att_rating}, DEF={self.def_rating}>"